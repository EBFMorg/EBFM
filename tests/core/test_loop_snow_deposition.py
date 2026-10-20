# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5.5

"""
Snowfall and deposition in LOOP_SNOW.
"""

import unittest

import numpy as np

from ebfm.core.config import ColumnDiscretizationConfig

NL = 20
MAX_SUBZ = 0.1
# Thickness of the top ice layer (m) for which subD * subZ / subZ != subD: a
# deposition update with zero snow would round its density.
_TOP_SUBZ = 0.076
# 0.1 m w.e. at Dfreshsnow = 350 kg m-3 ("firn_only") is 0.29 m of fresh snow:
# three deposition passes of at most MAX_SUBZ each.
_SNOW = 0.1


def _make_case(C, snow):
    """Cold, isothermal ice columns, one per entry of `snow` (m w.e.)."""
    gpsum = len(snow)
    T = C["T0"] - 10.0
    # doubledepth off means no layer merging/splitting.
    column = ColumnDiscretizationConfig(nl=NL, max_subZ=MAX_SUBZ, doubledepth=False, split=(15,))
    grid = {"mask": np.ones(gpsum, dtype=int)}
    OUT = {
        "subZ": np.full((gpsum, NL), MAX_SUBZ),
        "subT": np.full((gpsum, NL), T),
        "subTmean": np.full((gpsum, NL), T),
        "subD": np.full((gpsum, NL), C["Dice"]),
        "subW": np.zeros((gpsum, NL)),
        "subS": np.zeros((gpsum, NL)),
        "surfH": np.zeros(gpsum),
        "Tsurf": np.full(gpsum, T),
        # Snowfall is the only mass input.
        "melt": np.zeros(gpsum),
        "moist_deposition": np.zeros(gpsum),
        "moist_sublimation": np.zeros(gpsum),
        "moist_condensation": np.zeros(gpsum),
        "moist_evaporation": np.zeros(gpsum),
        "runoff_irr_deep_mean": np.zeros(gpsum),
    }
    OUT["subZ"][:, 0] = _TOP_SUBZ
    yearsnow = np.full(gpsum, 400.0)
    IN = {
        "T": np.full(gpsum, T),
        "WS": np.full(gpsum, 3.0),
        "snow": np.array(snow, dtype=float),
        "rain": np.zeros(gpsum),
        "yearsnow": np.tile(yearsnow[:, None], (1, NL)),
        "logyearsnow": np.tile(np.log(yearsnow)[:, None], (1, NL)),
    }
    return grid, column, OUT, IN


class TestSnowfallAndDeposition(unittest.TestCase):
    def setUp(self):
        from ebfm.core import INIT, LOOP_SNOW

        self.LOOP_SNOW = LOOP_SNOW
        self.C = INIT.init_constants()

    def _run(self, snow):
        """Advance the case by one hourly time step and return the state after."""
        grid, column, OUT, IN = _make_case(self.C, snow)
        phys = {"snow_compaction": "firn_only", "percolation": "bucket"}
        self.LOOP_SNOW.main(self.C, OUT, IN, 1.0 / 24.0, grid, phys, column)
        return OUT

    def test_column_without_snowfall_is_independent_of_its_neighbours(self):
        """A column without snowfall comes out the same, bit for bit, alone and next to a snowy column.

        The snowy column needs several deposition passes, the column without snowfall none.
        """
        # Preconditions for the case to detect a deposition update with zero snow.
        self.assertNotEqual(self.C["Dice"] * _TOP_SUBZ / _TOP_SUBZ, self.C["Dice"])
        self.assertGreater(_SNOW * self.C["Dwater"] / self.C["Dfreshsnow"], 2 * MAX_SUBZ)

        alone = self._run([0.0])
        with_neighbour = self._run([0.0, _SNOW])

        for name in ("subZ", "subD", "subT"):
            with self.subTest(name=name):
                np.testing.assert_array_equal(with_neighbour[name][0], alone[name][0])


if __name__ == "__main__":
    unittest.main()
