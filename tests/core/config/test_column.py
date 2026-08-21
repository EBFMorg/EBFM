# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5

"""The column discretization configuration and the invariants it enforces.

The parameters are hardcoded, so these tests serve two purposes: they pin down that the shipped defaults are
self-consistent, and they check that each guard actually fires, since a guard that never triggers is no guard at all.
"""

import unittest

import numpy as np

from ebfm.core.config import ColumnDiscretizationConfig


class TestShippedDefaults(unittest.TestCase):
    def test_defaults_are_accepted(self):
        """The shipped values must pass their own validation."""
        ColumnDiscretizationConfig()  # must not raise

    def test_defaults_leave_room_for_the_layer_zones(self):
        """Every split entry must address a layer that exists, with room on both sides, because
        layer_merging_and_splitting indexes subZ[:, split] and subZ[:, split - 2]."""
        config = ColumnDiscretizationConfig()

        self.assertGreaterEqual(config.split[0], 2)
        self.assertLessEqual(config.split[-1], config.nl - 1)
        self.assertTrue(np.all(np.diff(config.split) > 0), f"split not increasing: {config.split}")

    def test_deepest_layer_thickness_follows_from_the_split_count(self):
        """The documented relation between max_subZ, split and the deepest zone: LOOP_SNOW.melt_sublimation rebuilds
        the base layer as 2 ** len(split) * max_subZ, so this is the thickness the config implies."""
        config = ColumnDiscretizationConfig()

        self.assertEqual(2 ** len(config.split) * config.max_subZ, 0.8)


class TestInvariants(unittest.TestCase):
    """Each guard is exercised by constructing a config that violates it."""

    def test_too_few_layers_is_rejected(self):
        for nl in (0, 1, 2):
            with self.subTest(nl=nl), self.assertRaises(ValueError):
                ColumnDiscretizationConfig(nl=nl, split=(2,))

    def test_non_positive_top_layer_thickness_is_rejected(self):
        for max_subZ in (0.0, -0.1):
            with self.subTest(max_subZ=max_subZ), self.assertRaises(ValueError):
                ColumnDiscretizationConfig(max_subZ=max_subZ)

    def test_non_increasing_split_is_rejected(self):
        for split in ((25, 15), (15, 15)):
            with self.subTest(split=split), self.assertRaises(ValueError):
                ColumnDiscretizationConfig(split=split)

    def test_empty_split_is_rejected(self):
        with self.assertRaises(ValueError):
            ColumnDiscretizationConfig(split=())

    def test_split_too_close_to_the_surface_is_rejected(self):
        """split - 2 would wrap around to the deepest layer."""
        for split in ((0, 25), (1, 25)):
            with self.subTest(split=split), self.assertRaises(ValueError):
                ColumnDiscretizationConfig(split=split)

    def test_split_below_the_deepest_layer_is_rejected(self):
        """subZ[:, split] would be out of bounds."""
        with self.assertRaises(ValueError):
            ColumnDiscretizationConfig(nl=50, split=(15, 50))

    def test_split_on_the_deepest_layer_is_accepted(self):
        """nl - 1 is the last valid index, so it must not be rejected."""
        ColumnDiscretizationConfig(nl=50, split=(15, 49))  # must not raise


if __name__ == "__main__":
    unittest.main()
