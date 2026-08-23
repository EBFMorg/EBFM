# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np

from ebfm.core import LOOP_EBM_icon_land
from ebfm.core.constants import SECONDS_PER_DAY


class _FakeCoupler:
    def __init__(self, coupled: bool):
        self._coupled = coupled

    def has_coupling_to(self, name: str) -> bool:
        return self._coupled and name == "icon_land"


class TestPartitionEvapotrans(unittest.TestCase):
    T0 = 273.15

    def test_partition(self):
        # frozen surface: loss -> sublimation, gain -> deposition
        # melting surface: loss -> evaporation (limited by melt), gain -> condensation
        Tsurf = np.array([250.0, 250.0, 273.15, 273.15, 273.15])
        evapotrans = np.array([-1e-3, 2e-3, -3e-3, 4e-3, -5e-3])
        melt = np.array([0.0, 0.0, 1e-2, 0.0, 1e-3])

        moist = LOOP_EBM_icon_land.partition_evapotrans(evapotrans, Tsurf, melt, self.T0)

        np.testing.assert_allclose(moist["moist_sublimation"], [1e-3, 0, 0, 0, 0])
        np.testing.assert_allclose(moist["moist_deposition"], [0, 2e-3, 0, 0, 0])
        np.testing.assert_allclose(moist["moist_evaporation"], [0, 0, 3e-3, 0, 1e-3])  # last one limited by melt
        np.testing.assert_allclose(moist["moist_condensation"], [0, 0, 0, 4e-3, 0])


class TestIconLandEnergyBalance(unittest.TestCase):
    C = {"T0": 273.15, "Lm": 0.33e6, "dayseconds": SECONDS_PER_DAY, "eps": 0.98, "boltz": 5.67e-8}

    def test_is_available(self):
        IN = {"lice_t_srf": np.zeros(2), "lice_melt": np.zeros(2), "lice_evapotrans": np.zeros(2)}
        self.assertTrue(LOOP_EBM_icon_land.is_available(IN, _FakeCoupler(True)))
        self.assertFalse(LOOP_EBM_icon_land.is_available(IN, _FakeCoupler(False)))
        del IN["lice_melt"]
        self.assertFalse(LOOP_EBM_icon_land.is_available(IN, _FakeCoupler(True)))

    def test_main(self):
        dt = 0.125  # days
        IN = {
            "lice_t_srf": np.array([260.0, 274.0]),  # second value above melting point -> capped
            "lice_melt": np.array([0.0, 2e-3]),
            "lice_evapotrans": np.array([-1e-4, -5e-3]),
        }
        # results of EBFM's own energy balance (as computed by LOOP_EBM before the call)
        OUT = {
            "Tsurf": np.array([255.0, 270.0]),
            "melt": np.array([1e-4, 3e-3]),
            "Emelt": np.array([1.0, 30.0]),
            "moist_sublimation": np.array([2e-4, 0.0]),
            "moist_evaporation": np.array([0.0, 1e-4]),
            "moist_deposition": np.zeros(2),
            "moist_condensation": np.zeros(2),
            "SHF": np.array([10.0, 20.0]),
        }

        OUT = LOOP_EBM_icon_land.main(self.C, OUT, IN, {"dt": dt})

        # ICON-Land's results drive the model ...
        np.testing.assert_allclose(OUT["Tsurf"], [260.0, 273.15])
        np.testing.assert_allclose(OUT["melt"], [0.0, 2e-3])
        np.testing.assert_allclose(OUT["moist_sublimation"], [1e-4, 0.0])
        np.testing.assert_allclose(OUT["moist_evaporation"], [0.0, 2e-3])  # limited by melt
        np.testing.assert_allclose(OUT["Emelt"], [0.0, 2e-3 * 1e3 * 0.33e6 / (SECONDS_PER_DAY * dt)])
        # ... EBFM's own are kept as diagnostics ...
        np.testing.assert_allclose(OUT["ebm_Tsurf"], [255.0, 270.0])
        np.testing.assert_allclose(OUT["ebm_melt"], [1e-4, 3e-3])
        np.testing.assert_allclose(OUT["ebm_Emelt"], [1.0, 30.0])
        np.testing.assert_allclose(OUT["ebm_moist_sublimation"], [2e-4, 0.0])
        # ... and the flux diagnostics of EBFM's own balance are untouched
        np.testing.assert_allclose(OUT["SHF"], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
