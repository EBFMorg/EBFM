# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5

"""The vertical grid shifts in LOOP_SNOW.

Two blocks move the subsurface column by one layer index: snowfall_and_deposition
pushes it down when a new top layer is started, and melt_sublimation pulls it up
when the top layer is gone. Both must move *every* interior layer, and the layer
that drops off the bottom is the one snowfall_and_deposition credits to
runoff_irr_deep as water leaving the domain.

These tests use a column whose per-layer values are all distinct, and in
particular whose two deepest layers differ. That is essential: the model's own
initial state gives every deep layer the same thickness, temperature, density and
water content (INIT sets subD = C["Dice"] throughout and subZ[34:] = 0.8), and
duplicating or dropping a layer identical to its neighbour is numerically
invisible. A test built on a uniform deep column passes whether the shifts are
right or wrong.

subW is the tracer of choice, because in this setup nothing but the shift can
touch it: the column sits exactly at the melting point so it has no cold content
and no water can refreeze, no rain or melt adds water, and every layer's water
stays below the mliqmax cap that compaction applies. That makes the assertions on
subW exact rather than approximate.
"""

import unittest

import numpy as np

GPSUM, NL = 2, 50
# Layer thickness (m). The top layer is a full max_subZ, so any snowfall starts a
# new one and forces a downward shift.
MAX_SUBZ = 0.1
THICKNESS = 0.5
# Distinct per layer, and inside [C["Dfirn"], C["Dice"]) so only firn compaction
# applies. The 3.0 kg m-3 spacing is what makes a misplaced layer detectable.
DENSITY_STEP = 3.0
DENSITY_BASE = 600.0
# Distinct per layer, and far below the mliqmax of these layers (9.6 kg m-2 at the
# base rising to 13.5 at the top), so compaction's cap leaves them untouched.
WATER_BASE, WATER_STEP = 1.0, 0.1


def _make_case(C):
    """A column set at the melting point whose every layer differs from its neighbours."""
    subZ = np.full(NL, THICKNESS)
    subZ[0] = MAX_SUBZ
    subD = DENSITY_BASE + DENSITY_STEP * np.arange(NL)
    subW = WATER_BASE + WATER_STEP * np.arange(NL)

    grid = {
        "gpsum": GPSUM,
        "nl": NL,
        "max_subZ": MAX_SUBZ,
        "doubledepth": False,
        "split": np.array([15]),
        # Layer merging/splitting does its own shifts and its own runoff_irr_deep
        # credit; switch it off so these tests see the snowfall/melt shifts alone.
        "mask": np.zeros(GPSUM, dtype=int),
    }
    OUT = {
        "subZ": np.tile(subZ, (GPSUM, 1)),
        "subD": np.tile(subD, (GPSUM, 1)),
        "subW": np.tile(subW, (GPSUM, 1)),
        "subT": np.full((GPSUM, NL), C["T0"]),
        "subTmean": np.full((GPSUM, NL), C["T0"]),
        "subS": np.zeros((GPSUM, NL)),
        "surfH": np.zeros(GPSUM),
        "Tsurf": np.full(GPSUM, C["T0"]),
        "melt": np.zeros(GPSUM),
        "moist_deposition": np.zeros(GPSUM),
        "moist_sublimation": np.zeros(GPSUM),
        "moist_condensation": np.zeros(GPSUM),
        "moist_evaporation": np.zeros(GPSUM),
        "runoff_irr_deep_mean": np.zeros(GPSUM),
    }
    yearsnow = np.full(GPSUM, 500.0)
    IN = {
        "T": np.full(GPSUM, C["T0"]),
        "WS": np.full(GPSUM, 4.0),
        "snow": np.zeros(GPSUM),
        "rain": np.zeros(GPSUM),
        "yearsnow": np.tile(yearsnow[:, None], (1, NL)),
        "logyearsnow": np.tile(np.log(yearsnow)[:, None], (1, NL)),
    }
    return grid, OUT, IN


class TestColumnShift(unittest.TestCase):
    def setUp(self):
        from ebfm.core import INIT, LOOP_SNOW

        self.LOOP_SNOW = LOOP_SNOW
        self.C = INIT.init_constants()
        self.phys = {"snow_compaction": "firn+snow", "percolation": "normal"}

    def _step(self, grid, OUT, IN):
        self.LOOP_SNOW.main(self.C, OUT, IN, 1.0 / 24.0, grid, self.phys)

    def _snowfall_case(self):
        """A case that performs exactly one downward shift.

        The fresh snow is sized from the fresh-snow density the model will compute, so
        that it overfills the top layer without filling a second one: one pass of the
        shift loop, whose result is a plain permutation of the column.
        """
        grid, OUT, IN = _make_case(self.C)
        Dfresh = 50 + 1.7 * 17 ** (3 / 2) + 266.86 * (0.5 * (1 + np.tanh(4.0 / 5))) ** 8.8
        IN["snow"][:] = 0.9 * MAX_SUBZ * Dfresh / self.C["Dwater"]
        return grid, OUT, IN

    def _sublimation_case(self):
        """A case that performs exactly one upward shift.

        Sublimation, not melt, removes the top layer: it takes away exactly as much mass
        as the top layer holds, and unlike melt it produces no liquid water, so subW
        stays an exact tracer. Both drive the same block through the same `shift_tot`.
        """
        grid, OUT, IN = _make_case(self.C)
        OUT["moist_sublimation"][:] = OUT["subD"][:, 0] * OUT["subZ"][:, 0] * 1e-3
        return grid, OUT, IN

    def test_snowfall_shifts_every_layer_down(self):
        """Every layer moves down one index; the deepest layer leaves the domain.

        The layer that must not survive is the deepest one: if it is still there
        afterwards, the column kept a layer that the runoff bookkeeping already
        counted as gone.
        """
        grid, OUT, IN = self._snowfall_case()
        subW_old = OUT["subW"].copy()

        self._step(grid, OUT, IN)

        want = np.empty((GPSUM, NL))
        want[:, 0] = 0.0  # new top layer is fresh snow, no water
        want[:, 1] = subW_old[:, 0]  # former top layer, handed down whole
        want[:, 2:NL] = subW_old[:, 1 : NL - 1]  # everything else moves down one index
        np.testing.assert_array_equal(
            OUT["subW"],
            want,
            err_msg="subW is not the column shifted down by one index",
        )

    def test_sublimation_shifts_every_layer_up(self):
        """Every layer moves up one index; a fresh, dry layer appears at the base."""
        grid, OUT, IN = self._sublimation_case()
        subW_old = OUT["subW"].copy()

        self._step(grid, OUT, IN)

        want = np.empty((GPSUM, NL))
        want[:, 0] = subW_old[:, 1]  # former layer 1 becomes the whole top layer
        want[:, 1 : NL - 1] = subW_old[:, 2:NL]  # everything else moves up one index
        want[:, NL - 1] = 0.0  # fresh layer added at the base
        np.testing.assert_allclose(
            OUT["subW"],
            want,
            rtol=1e-12,
            atol=0.0,
            err_msg="subW is not the column shifted up by one index",
        )

    def test_snowfall_shifts_density_down(self):
        """The density profile rides the same shift as the water profile.

        Firn compaction perturbs subD by ~1e-3 kg m-3 over one hourly step, three
        orders of magnitude below the layer-to-layer spacing, so a misplaced layer is
        unambiguous at this tolerance.
        """
        grid, OUT, IN = self._snowfall_case()
        subD_old = OUT["subD"].copy()

        self._step(grid, OUT, IN)

        np.testing.assert_allclose(
            OUT["subD"][:, 2:NL],
            subD_old[:, 1 : NL - 1],
            rtol=1e-4,
            err_msg="subD is not the column shifted down by one index",
        )

    def test_sublimation_shifts_density_up(self):
        """The density profile rides the same shift as the water profile."""
        grid, OUT, IN = self._sublimation_case()
        subD_old = OUT["subD"].copy()

        self._step(grid, OUT, IN)

        np.testing.assert_allclose(
            OUT["subD"][:, 1 : NL - 1],
            subD_old[:, 2:NL],
            rtol=1e-4,
            err_msg="subD is not the column shifted up by one index",
        )

    def test_snowfall_shift_conserves_column_mass(self):
        """Column mass changes by exactly the snow added minus the layer that left.

        Nothing else in this configuration moves mass: compaction preserves subD*subZ
        by construction, heat conduction touches neither, and with no rain, no melt and
        no cold content there is no refreezing. So the whole residual is the shift's.
        """
        grid, OUT, IN = self._snowfall_case()
        mass_before = np.sum(OUT["subD"] * OUT["subZ"], axis=1)
        # The fresh snow mass is the snowfall in water equivalent, whatever density the
        # model gives it: shift_snowfall * Dfreshsnow == snow * Dwater.
        mass_added = IN["snow"] * self.C["Dwater"]
        mass_left = OUT["subD"][:, NL - 1] * OUT["subZ"][:, NL - 1]

        self._step(grid, OUT, IN)

        mass_after = np.sum(OUT["subD"] * OUT["subZ"], axis=1)
        np.testing.assert_allclose(
            mass_after - mass_before,
            mass_added - mass_left,
            atol=1e-6,
            err_msg="column mass across the shift does not balance snowfall in minus deepest layer out",
        )

    def test_water_credited_to_deep_runoff_actually_leaves(self):
        """runoff_irr_deep must be credited the water that really left the column.

        This is the bookkeeping the shift has to agree with: the block credits the
        deepest layer's water as having left the domain, so the deepest layer's water
        is what the shift has to discard.
        """
        grid, OUT, IN = self._snowfall_case()
        water_before = np.sum(OUT["subW"], axis=1)

        self._step(grid, OUT, IN)

        water_after = np.sum(OUT["subW"], axis=1)
        # runoff() has already rescaled runoff_irr_deep into the running mean, so read
        # the credit back off the mean, which started at zero.
        weight = (1.0 / 24.0) / self.C["yeardays"]
        credited = OUT["runoff_irr_deep_mean"] / weight

        np.testing.assert_allclose(
            credited,
            water_before - water_after,
            rtol=1e-12,
            err_msg="water credited to runoff_irr_deep differs from the water that left the column",
        )


if __name__ == "__main__":
    unittest.main()
