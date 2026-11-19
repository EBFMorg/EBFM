# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5.5

"""The checks of snowfall and melt in LOOP_SNOW.

Corrupted input (implausible snowfall or melt, or a layer that melt removal
cannot get through) must stop the run with a message naming
the column and its location, instead of hanging it in a grid-shift loop.
Extreme but physical input must pass. All backends share this code.
"""

import unittest
import warnings

import numpy as np

from ebfm.core import INIT

from test_loop_snow_heat_conduction_checks import BAD_COLUMN, make_case, step

# name: ({(dict, field, index): value}, expected message fragments)
CORRUPTIONS = {
    "huge snowfall": (
        {("IN", "snow", BAD_COLUMN): 1e6},
        ["thicker than 1000 top layers", "snow=1e+06"],
    ),
    "infinite snowfall": (
        {("IN", "snow", BAD_COLUMN): np.inf},
        ["thicker than 1000 top layers", "snow=inf"],
    ),
    "melt beyond the column": (
        {("OUT", "melt", BAD_COLUMN): 1e6},
        ["more mass than the whole column holds", "melt=1e+06"],
    ),
    "near-zero layer below the melting surface": (
        {("OUT", "subZ", (BAD_COLUMN, 1)): 1e-30, ("OUT", "melt", BAD_COLUMN): 0.2},
        ["did not finish within 1000 grid-shift passes", "subZ[1]=1e-30 m"],
    ),
}

# name: {(dict, field, index): value}
EXTREME_BUT_PHYSICAL = {
    "heavy snowfall": {("IN", "snow", BAD_COLUMN): 1.0},
    "strong melt": {("OUT", "melt", BAD_COLUMN): 0.1},
}


def apply(changes, OUT, IN):
    dicts = {"OUT": OUT, "IN": IN}
    for (name, field, index), value in changes.items():
        dicts[name][field][index] = value


class TestSnowfallMeltChecks(unittest.TestCase):
    def setUp(self):
        self.C = INIT.init_constants()
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def test_corrupted_input_is_reported(self):
        for name, (changes, fragments) in CORRUPTIONS.items():
            with self.subTest(case=name):
                grid, OUT, IN = make_case(self.C)
                apply(changes, OUT, IN)
                with self.assertRaises(AssertionError) as raised:
                    step(self.C, grid, OUT, IN)
                message = str(raised.exception)
                self.assertIn("in 1 of 2 columns", message)
                self.assertIn("column 1 (lat=71.0000, lon=321.0000)", message)
                for fragment in fragments:
                    self.assertIn(fragment, message)

    def test_extreme_but_physical_input_passes(self):
        for name, changes in EXTREME_BUT_PHYSICAL.items():
            with self.subTest(case=name):
                grid, OUT, IN = make_case(self.C)
                apply(changes, OUT, IN)
                step(self.C, grid, OUT, IN)


if __name__ == "__main__":
    unittest.main()
