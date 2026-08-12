# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5

"""
Smoke test for the "uniform" percolation scheme of LOOP_SNOW.

The scheme distributes the available water evenly over the layers down to the
characteristic percolation depth C["perc_depth"] and adds nothing below it.
That cut-off layer is determined per column, so columns with different layer
thicknesses wet down to different layer indices.

The default configuration uses percolation="normal" and the scheme is not
reachable from the command line, so this code path had never been executed.
"""

import unittest

import numpy as np

GPSUM, NL = 4, 20
# Layer thickness per column (m), constant over depth. With perc_depth = 6.0 m
# the layer midpoints zz = cumsum(subZ) - subZ/2 put the cut-off at
#   0.7 m layers: zz[8] = 5.95 (next one 6.65) -> layer 8
#   1.3 m layers: zz[4] = 5.85 (next one 7.15) -> layer 4
_THICKNESS = np.array([0.7, 0.7, 1.3, 1.3])
_EXPECTED_CUTOFF = np.array([8, 8, 4, 4])


def _make_case(C):
    """A cold, dry, uniformly layered column set whose only water input is rain."""
    grid = {
        "gpsum": GPSUM,
        "nl": NL,
        "max_subZ": 1.5,
        # No layer merging/splitting, so the grid stays exactly as built here.
        "doubledepth": False,
        "split": np.array([15]),
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
    return grid, OUT, IN


class TestUniformPercolation(unittest.TestCase):
    """percolation="uniform" must run and must stop at the percolation depth."""

    def setUp(self):
        from ebfm.core import INIT, LOOP_SNOW

        self.LOOP_SNOW = LOOP_SNOW
        self.C = INIT.init_constants()
        self.phys = {"snow_compaction": "firn+snow", "percolation": "uniform"}
        self.grid, self.OUT, self.IN = _make_case(self.C)

    def test_uniform_percolation_runs(self):
        """The scheme used to raise TypeError on the first timestep.

        The layer range was selected with a slice built from the per-column
        index array `ind`, which NumPy cannot use as a slice bound.
        """
        self.LOOP_SNOW.main(self.C, self.OUT, self.IN, 1.0 / 24.0, self.grid, self.phys)

    def test_uniform_percolation_stops_at_percolation_depth(self):
        """Water reaches the cut-off layer of its own column, and no layer below.

        The incoming rain refreezes in the cold layers it reaches, which raises
        their temperature; layers that receive no water keep theirs. The two
        column groups differ only in layer thickness, so a cut-off shared by all
        columns cannot reproduce this.
        """
        subT_before = self.OUT["subT"].copy()
        self.LOOP_SNOW.main(self.C, self.OUT, self.IN, 1.0 / 24.0, self.grid, self.phys)

        warmed = self.OUT["subT"] > subT_before
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
