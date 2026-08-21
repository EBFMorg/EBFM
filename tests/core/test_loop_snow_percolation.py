# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5

"""
The percolation schemes in LOOP_SNOW.

"bucket", "normal" and "linear".
"""

import unittest

import numpy as np

from ebfm.core.config import ColumnDiscretizationConfig

GPSUM, NL = 4, 20
_SCHEMES = ("bucket", "normal", "linear")
# Layer thickness per column (m), constant over depth: the columns differ in thickness only.
_THICKNESS = np.array([0.7, 0.7, 1.3, 1.3])
# Deepest layer still reached by percolation="linear": the last one whose midpoint
# zz = cumsum(subZ) - subZ/2 stays above perc_depth = 6.0 m (0.7 m: zz[8] = 5.95, 1.3 m: zz[4] = 5.85).
_EXPECTED_CUTOFF = np.array([8, 8, 4, 4])


def _make_case(C):
    """A cold, dry, uniformly layered column set whose only water input is rain."""
    # max_subZ is above _THICKNESS so nothing shifts, and doubledepth off means no
    # layer merging/splitting: the column stays exactly as built here.
    column = ColumnDiscretizationConfig(nl=NL, max_subZ=1.5, doubledepth=False, split=(15,))
    grid = {
        "gpsum": GPSUM,
        "mask": np.ones(GPSUM, dtype=int),
    }
    OUT = {
        "subZ": np.repeat(_THICKNESS[:, None], NL, axis=1),
        "subT": np.full((GPSUM, NL), C["T0"] - 10.0),
        "subTmean": np.full((GPSUM, NL), C["T0"] - 10.0),
        "subD": np.full((GPSUM, NL), 400.0),
        "subW": np.zeros((GPSUM, NL)),
        "subS": np.zeros((GPSUM, NL)),
        "surfH": np.zeros(GPSUM),
        "Tsurf": np.full(GPSUM, C["T0"] - 10.0),
        # No melt, sublimation or deposition: the vertical grid is not shifted,
        # so percolation is the only process that moves water down the column.
        "melt": np.zeros(GPSUM),
        "moist_deposition": np.zeros(GPSUM),
        "moist_sublimation": np.zeros(GPSUM),
        "moist_condensation": np.zeros(GPSUM),
        "moist_evaporation": np.zeros(GPSUM),
        "runoff_irr_deep_mean": np.zeros(GPSUM),
    }
    yearsnow = np.full(GPSUM, 400.0)
    IN = {
        "T": np.full(GPSUM, C["T0"] - 10.0),
        "WS": np.full(GPSUM, 3.0),
        "snow": np.zeros(GPSUM),
        "rain": np.full(GPSUM, 5e-3),  # the only water input
        "yearsnow": np.tile(yearsnow[:, None], (1, NL)),
        "logyearsnow": np.tile(np.log(yearsnow)[:, None], (1, NL)),
    }
    return grid, column, OUT, IN


class TestPercolationSchemes(unittest.TestCase):
    def setUp(self):
        from ebfm.core import INIT, LOOP_SNOW

        self.LOOP_SNOW = LOOP_SNOW
        self.C = INIT.init_constants()

    def _run(self, percolation):
        """Advance the case by one hourly time step and return the state before and after."""
        grid, column, OUT, IN = _make_case(self.C)
        phys = {"snow_compaction": "firn+snow", "percolation": percolation}
        subT_before = OUT["subT"].copy()
        self.LOOP_SNOW.main(self.C, OUT, IN, 1.0 / 24.0, grid, phys, column)
        return subT_before, OUT

    def test_supported_schemes_run(self):
        """One time step must complete and leave a finite state for every scheme."""
        for percolation in _SCHEMES:
            with self.subTest(percolation=percolation):
                _, OUT = self._run(percolation)
                for name in ("subT", "subD", "subZ", "subW", "subS"):
                    self.assertTrue(np.isfinite(OUT[name]).all(), f"{name} is not finite")

    def test_rain_warms_the_column(self):
        """The rain refreezes in the cold layers it reaches, so every scheme warms the column.

        The column set is isothermal, so refreezing is the only process that changes subT.
        """
        for percolation in _SCHEMES:
            with self.subTest(percolation=percolation):
                subT_before, OUT = self._run(percolation)
                warmed = OUT["subT"] > subT_before
                self.assertTrue(warmed[:, 0].all(), f"surface layer stayed cold, got {warmed}")
                self.assertTrue((OUT["subT"] <= self.C["T0"]).all(), "refreezing warmed a layer above T0")

    def test_bucket_percolation_wets_only_the_surface_layer(self):
        """All water enters the surface layer, which here holds it all: no layer below warms."""
        subT_before, OUT = self._run("bucket")

        warmed = OUT["subT"] > subT_before
        self.assertTrue(warmed[:, 0].all(), f"surface layer should have received water, got {warmed}")
        self.assertFalse(warmed[:, 1:].any(), f"no layer below the surface should have received water, got {warmed}")

    def test_linear_percolation_stops_at_percolation_depth(self):
        """Water reaches the cut-off layer of its own column, and no layer below.

        The cut-off follows from the layer thickness, so the thin and the thick columns
        stop at different layers: a cut-off shared by all columns cannot reproduce this.
        """
        subT_before, OUT = self._run("linear")

        warmed = OUT["subT"] > subT_before
        for column, cutoff in enumerate(_EXPECTED_CUTOFF):
            with self.subTest(column=column, thickness=_THICKNESS[column]):
                self.assertTrue(
                    warmed[column, : cutoff + 1].all(),
                    f"column {column}: layers 0..{cutoff} should have received water, got {warmed[column]}",
                )
                self.assertFalse(
                    warmed[column, cutoff + 1 :].any(),
                    f"column {column}: no layer below {cutoff} should have received water, got {warmed[column]}",
                )

        # The point of the per-column cut-off: thin and thick columns differ.
        self.assertNotEqual(_EXPECTED_CUTOFF[0], _EXPECTED_CUTOFF[-1])


if __name__ == "__main__":
    unittest.main()
