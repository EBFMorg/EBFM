# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
The percolation schemes LOOP_SNOW accepts.

"bucket", "normal" and "linear" run. Anything else (including the removed
"uniform" scheme) is rejected with a clear error instead of being silently
ignored.
"""

import unittest

import numpy as np

GPSUM, NL = 4, 20


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
        "subZ": np.full((GPSUM, NL), 1.0),
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


class TestPercolationSchemes(unittest.TestCase):
    def setUp(self):
        from ebfm.core import INIT, LOOP_SNOW

        self.LOOP_SNOW = LOOP_SNOW
        self.C = INIT.init_constants()

    def _run(self, percolation):
        grid, OUT, IN = _make_case(self.C)
        phys = {"snow_compaction": "firn+snow", "percolation": percolation}
        self.LOOP_SNOW.main(self.C, OUT, IN, 1.0 / 24.0, grid, phys)
        return OUT

    def test_supported_schemes_run(self):
        for percolation in ("bucket", "normal", "linear"):
            with self.subTest(percolation=percolation):
                OUT = self._run(percolation)
                self.assertTrue(np.isfinite(OUT["subT"]).all())

    def test_removed_and_unknown_schemes_are_rejected(self):
        """"uniform" was removed; selecting it must fail, not fall through."""
        for percolation in ("uniform", "not-a-scheme"):
            with self.subTest(percolation=percolation):
                with self.assertRaises(ValueError):
                    self._run(percolation)


if __name__ == "__main__":
    unittest.main()
