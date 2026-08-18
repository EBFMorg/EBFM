# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5

"""
Correctness gate for the GPU backend of LOOP_SNOW.

Runs the whole snow model on both the NumPy and the GPU backend from an
identical state and requires the results to agree to the same tolerance the
--dump-reference gate uses (atol=1e-12, rtol=1e-9).

No GPU is needed: numba's CUDA simulator (NUMBA_ENABLE_CUDASIM=1) executes the
@cuda.jit kernels on the CPU, one Python thread per grid column. This is slow,
so the grid here is deliberately tiny. It is a logic check, not a benchmark.
The simulator has to be enabled before numba is first imported, which is why
this module sets the environment variable at import time and is skipped when
numba was already imported without it.

Note that the simulator only checks kernel logic. Device-specific behaviour
(occupancy, races between blocks, driver/toolkit issues) is not covered here
and is verified with --dump-reference on real hardware.
"""

import os
import unittest
import warnings

import numpy as np

# Must happen before `from numba import cuda` anywhere in the process.
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

_CUDASIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

try:
    import numba  # noqa: F401

    _NUMBA = True
except ImportError:  # pragma: no cover, numba is an optional dependency
    _NUMBA = False


def _make_case(gpsum=48, nl=50, seed=7, snow_compaction="firn+snow", percolation="normal"):
    """Build a small but non-degenerate LOOP_SNOW state.

    The columns are randomised across the regimes the kernels branch on: snow
    and firn and solid ice densities, wet and dry layers, columns that gain
    enough snow to shift the grid down and columns that melt enough to shift it
    up, and layer thicknesses on both sides of the merge/split thresholds.

    `snow_compaction` and `percolation` select the physics branch the kernels
    take; see _SNOW_COMPACTION_MODES / _PERCOLATION_MODES below.
    """
    from ebfm.core import INIT

    rng = np.random.default_rng(seed)
    C = INIT.init_constants()

    grid = {
        "gpsum": gpsum,
        "nl": nl,
        "max_subZ": 0.1,
        "doubledepth": True,
        "split": np.array([15, 25, 35]),
        "mask": np.ones(gpsum, dtype=int),
    }
    # A few inactive columns, so the kernel's mask early-out is exercised.
    grid["mask"][:3] = 0

    phys = {"snow_compaction": snow_compaction, "percolation": percolation}
    dt = 1.0 / 24.0

    OUT = {}
    OUT["subT"] = C["T0"] - rng.uniform(0.0, 25.0, (gpsum, nl))
    OUT["subTmean"] = OUT["subT"] - rng.uniform(0.0, 5.0, (gpsum, nl))
    OUT["subD"] = rng.uniform(250.0, 900.0, (gpsum, nl))
    # Force a spread of regimes: pure snow, firn, and fully compacted ice.
    OUT["subD"][::4, :] = rng.uniform(250.0, 480.0, OUT["subD"][::4, :].shape)
    OUT["subD"][1::4, :] = C["Dice"]
    OUT["subZ"] = np.full((gpsum, nl), grid["max_subZ"])
    for n, split in enumerate(grid["split"]):
        OUT["subZ"][:, split - 1 :] = (2.0 ** (n + 1)) * grid["max_subZ"]
    OUT["subZ"][:, -1] = (2.0 ** len(grid["split"])) * grid["max_subZ"]
    # Push some columns over the merge threshold and some under the split one.
    OUT["subZ"][::5, grid["split"]] = grid["max_subZ"] * 0.5
    OUT["subZ"][2::5, np.array(grid["split"]) - 2] = grid["max_subZ"] * 8.0
    OUT["subW"] = rng.uniform(0.0, 3.0, (gpsum, nl))
    OUT["subW"][::3, :] = 0.0
    OUT["subS"] = rng.uniform(0.0, 2.0, (gpsum, nl))
    OUT["subS"][1::3, :] = 0.0
    OUT["surfH"] = rng.uniform(-1.0, 1.0, gpsum)
    OUT["Tsurf"] = C["T0"] - rng.uniform(0.0, 15.0, gpsum)
    # Conductivity / heat capacity diagnostics: INIT allocates these, and the
    # GPU path writes the kernel results back into them in place.
    OUT["subK"] = np.zeros((gpsum, nl))
    OUT["subCeff"] = np.zeros((gpsum, nl))

    # Surface energy balance results consumed by LOOP_SNOW.
    OUT["melt"] = rng.uniform(0.0, 0.02, gpsum)
    OUT["melt"][::6] = 0.0
    # A couple of columns melt hard enough to force several grid shifts.
    OUT["melt"][3::11] = 0.5
    OUT["moist_deposition"] = rng.uniform(0.0, 1e-4, gpsum)
    OUT["moist_sublimation"] = rng.uniform(0.0, 1e-4, gpsum)
    OUT["moist_condensation"] = rng.uniform(0.0, 1e-4, gpsum)
    OUT["moist_evaporation"] = rng.uniform(0.0, 1e-4, gpsum)
    OUT["runoff_irr_deep_mean"] = rng.uniform(0.0, 1.0, gpsum)
    # Deliberately NOT seeded here: "sumWinit", "cpi", "Dens_*" and "runoff_irr"
    # are not created by INIT either, LOOP_SNOW produces them. Seeding them
    # would hide a backend that only ever writes into an existing host array.

    IN = {}
    IN["T"] = C["T0"] - rng.uniform(-5.0, 25.0, gpsum)
    IN["WS"] = rng.uniform(0.5, 15.0, gpsum)
    IN["rain"] = rng.uniform(0.0, 5e-3, gpsum)
    IN["snow"] = rng.uniform(0.0, 3e-3, gpsum)
    # Some columns get enough snow for more than one grid shift.
    IN["snow"][2::9] = 0.3
    IN["snow"][::7] = 0.0
    ys = rng.uniform(100.0, 900.0, gpsum)
    IN["yearsnow"] = np.tile(ys[:, None], (1, nl))
    IN["logyearsnow"] = np.tile(np.log(ys)[:, None], (1, nl))

    return C, OUT, IN, dt, grid, phys


def _deepcopy_state(OUT, IN):
    copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in OUT.items()}
    copy_in = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in IN.items()}
    return copy, copy_in


# Fields both backends must agree on. The subsurface grids and every LOOP_SNOW
# output that leaves the module; --dump-reference compares a subset of these.
_COMPARED = [
    "subT",
    "subD",
    "subZ",
    "subW",
    "subS",
    "subTmean",
    "surfH",
    "T_ice",
    "runoff",
    "runoff_surf",
    "runoff_slush",
    "runoff_irr",
    "runoff_irr_deep",
    "refr",
    "refr_P",
    "refr_S",
    "refr_I",
    "slushw",
    "irrw",
    "Dens_destr_metam",
    "Dens_overb_pres",
    "Dens_drift",
]

# The physics options the kernels switch on: _compaction_kernel_gpu takes the
# snow_compaction branch, _percolation_kernel_gpu the percolation one. The
# default configuration exercises exactly one branch of each, so the modes are
# enumerated here instead.
_SNOW_COMPACTION_MODES = ("firn_only", "firn+snow")
_PERCOLATION_MODES = ("bucket", "normal", "linear", "uniform")


@unittest.skipUnless(_NUMBA and _CUDASIM, "requires numba with NUMBA_ENABLE_CUDASIM=1")
class TestLoopSnowGPUMatchesNumPy(unittest.TestCase):
    """The GPU backend must reproduce the NumPy backend, step for step."""

    @classmethod
    def setUpClass(cls):
        from ebfm.core import LOOP_SNOW, compute_backend

        cls.LOOP_SNOW = LOOP_SNOW
        cls.compute_backend = compute_backend

        # The drift-densification time scale divides by a gamma that underflows
        # to zero in the deepest layers, giving tau = inf and a zero increment.
        # The NumPy path wraps the same division in np.errstate(divide="ignore");
        # under the CUDA simulator it surfaces as a Python RuntimeWarning, which
        # a real device never raises.
        warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)

    def setUp(self):
        self.compute_backend._backend = self.compute_backend.ComputeBackend.NUMPY

    def tearDown(self):
        self.compute_backend._backend = self.compute_backend.ComputeBackend.NUMPY

    def _run(self, backend, steps, **case):
        """Run `steps` LOOP_SNOW timesteps on `backend` from a fixed state."""
        C, OUT, IN, dt, grid, phys = _make_case(**case)
        self.compute_backend._backend = backend
        for _ in range(steps):
            # Fresh copies of the per-step forcing, so both backends see the
            # same inputs even though LOOP_SNOW overwrites some of them.
            _, IN_step = _deepcopy_state(OUT, IN)
            self.LOOP_SNOW.main(C, OUT, IN_step, dt, grid, phys)
        return OUT

    def _assert_matches(self, ref, got, context):
        for key in _COMPARED:
            with self.subTest(field=key, context=context):
                self.assertIn(key, ref, f"{key} missing from the NumPy result")
                self.assertIn(key, got, f"{key} missing from the GPU result")
                a = np.asarray(ref[key], dtype=np.float64)
                b = np.asarray(got[key], dtype=np.float64)
                self.assertEqual(a.shape, b.shape, f"{key}: shape mismatch")
                if not np.allclose(a, b, atol=1e-12, rtol=1e-9, equal_nan=True):
                    diff = np.abs(a - b)
                    worst = np.unravel_index(np.nanargmax(diff), diff.shape)
                    self.fail(
                        f"{key} differs ({context}): max|diff|={np.nanmax(diff):.6e} "
                        f"at {worst}, numpy={a[worst]!r}, gpu={b[worst]!r}"
                    )

    def test_single_step_matches_numpy(self):
        ref = self._run(self.compute_backend.ComputeBackend.NUMPY, steps=1)
        self.setUp()
        got = self._run(self.compute_backend.ComputeBackend.GPU, steps=1)
        self._assert_matches(ref, got, "1 step")

    def test_multi_step_matches_numpy(self):
        """Three steps: catches state that is not carried correctly on the device."""
        ref = self._run(self.compute_backend.ComputeBackend.NUMPY, steps=3)
        self.setUp()
        got = self._run(self.compute_backend.ComputeBackend.GPU, steps=3)
        self._assert_matches(ref, got, "3 steps")

    def test_physics_modes_match_numpy(self):
        """Every snow_compaction / percolation branch of the kernels, one step each.

        The other tests only ever run "firn+snow" and "normal", which leaves the
        remaining kernel branches unexecuted.
        """
        for snow_compaction in _SNOW_COMPACTION_MODES:
            for percolation in _PERCOLATION_MODES:
                with self.subTest(snow_compaction=snow_compaction, percolation=percolation):
                    case = {"snow_compaction": snow_compaction, "percolation": percolation}
                    self.setUp()
                    ref = self._run(self.compute_backend.ComputeBackend.NUMPY, steps=1, **case)
                    self.setUp()
                    got = self._run(self.compute_backend.ComputeBackend.GPU, steps=1, **case)
                    self._assert_matches(ref, got, f"{snow_compaction}/{percolation}, 1 step")


if __name__ == "__main__":
    unittest.main()
