# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5.5

"""The input checks of heat conduction in LOOP_SNOW.

A corrupted column (temperatures in deg C, fill values, NaN, zero-thickness
layers) must stop the run with a message naming the column, its location and the
offending layers. A cold but physical column must pass.

test_loop_snow_gpu.py runs the same cases on the GPU backend.
"""

import unittest
import warnings

import numpy as np

from ebfm.core import INIT, LOOP_SNOW, LOOP_SNOW_checks, compute_backend
from ebfm.core.config import ColumnDiscretizationConfig

from test_loop_snow_layer_shift import _make_case

BAD_COLUMN, BAD_LAYER = 1, 10

# name: (field, index, value, expected message fragments)
CORRUPTIONS = {
    "temperature in deg C": (
        "subT",
        (BAD_COLUMN, BAD_LAYER),
        -30.0,
        ["subT < 200 K", f"layers [{BAD_LAYER}], min -30 K"],
    ),
    "fill value": ("subT", (BAD_COLUMN, BAD_LAYER), -9999.0, ["subT < 200 K", "min -9999 K"]),
    "implausibly cold": ("subT", (BAD_COLUMN, BAD_LAYER), 150.0, ["subT < 200 K", "min 150 K"]),
    "NaN temperature": (
        "subT",
        (BAD_COLUMN, BAD_LAYER),
        np.nan,
        ["Non-finite", f"subT non-finite in layers [{BAD_LAYER}]"],
    ),
    "NaN surface temperature": ("Tsurf", BAD_COLUMN, np.nan, ["Non-finite", "Tsurf=nan"]),
    "zero density": ("subD", (BAD_COLUMN, BAD_LAYER), 0.0, ["Non-finite", f"subD non-finite in layers [{BAD_LAYER}]"]),
    "zero thickness": ("subZ", (BAD_COLUMN, BAD_LAYER), 0.0, ["dt_stab <= 0", f"subZ <= 0 in layers [{BAD_LAYER}]"]),
}


def make_case(C):
    """Two healthy columns at the melting point, with a location for the messages."""
    grid, OUT, IN = _make_case(C)
    grid["lat"] = np.array([70.0, 71.0])
    grid["lon"] = np.array([320.0, 321.0])
    # Written back in place by the GPU backend.
    OUT["subK"] = np.zeros_like(OUT["subT"])
    OUT["subCeff"] = np.zeros_like(OUT["subT"])
    return grid, OUT, IN


def step(C, grid, OUT, IN):
    phys = {"snow_compaction": "firn+snow", "percolation": "normal"}
    LOOP_SNOW.main(C, OUT, IN, 1.0 / 24.0, grid, phys, ColumnDiscretizationConfig())


def assert_corruption_reported(test, C, field, index, value, fragments):
    grid, OUT, IN = make_case(C)
    OUT[field][index] = value
    with test.assertRaises(AssertionError) as raised:
        step(C, grid, OUT, IN)
    message = str(raised.exception)
    test.assertIn("in 1 of 2 columns", message)
    test.assertIn("column 1 (lat=71.0000, lon=321.0000)", message)
    for fragment in fragments:
        test.assertIn(fragment, message)


class TestHeatConductionChecks(unittest.TestCase):
    def setUp(self):
        self.C = INIT.init_constants()
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def tearDown(self):
        compute_backend._backend = compute_backend.ComputeBackend.NUMPY

    def _backends(self):
        backends = [compute_backend.ComputeBackend.NUMPY]
        if compute_backend._NUMBA_AVAILABLE:
            backends.append(compute_backend.ComputeBackend.NUMBA)
        return backends

    def test_corrupted_column_is_reported(self):
        for backend in self._backends():
            for name, (field, index, value, fragments) in CORRUPTIONS.items():
                with self.subTest(backend=backend.name, case=name):
                    compute_backend._backend = backend
                    assert_corruption_reported(self, self.C, field, index, value, fragments)

    def test_cold_physical_column_passes(self):
        for backend in self._backends():
            with self.subTest(backend=backend.name):
                compute_backend._backend = backend
                grid, OUT, IN = make_case(self.C)
                OUT["subT"][:, 1:] = 210.0
                step(self.C, grid, OUT, IN)

    def test_negative_density_is_named(self):
        grid, OUT, _ = make_case(self.C)
        OUT["subD"][BAD_COLUMN, BAD_LAYER] = -1.0
        dt_stab = np.array([1e-3, -1e-3])
        message = LOOP_SNOW_checks.describe_dt_stab_failure(dt_stab, OUT, grid)
        self.assertIn(f"subD <= 0 in layers [{BAD_LAYER}], min -1", message)

    def test_ghost_layer_temperature_is_not_checked(self):
        grid, OUT, _ = make_case(self.C)
        OUT["subT"][:, 0] = np.nan
        self.assertTrue(LOOP_SNOW_checks.heat_conduction_inputs_finite(OUT))


if __name__ == "__main__":
    unittest.main()
