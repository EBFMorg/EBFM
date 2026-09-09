# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Sonnet 5

"""
LOOP_EBM_GHF.conductance: the density-dependent conductance between the surface and the
midpoint of the second subsurface layer, used by LOOP_EBM_GHF.main to compute the ground
heat flux.
"""

import unittest

import numpy as np

from ebfm.core import LOOP_EBM_GHF


def _reference_conductance(subD, subZ, subT):
    """Point-by-point transcription of the formula in LOOP_EBM_GHF.conductance. Independent
    of the vectorized implementation, so an indexing or broadcasting slip there (e.g. mixing
    up layers 0/1, or an axis) shows up as a mismatch instead of being self-consistent."""
    gpsum, nl = subD.shape
    GHF_k = np.empty((gpsum, nl))
    for i in range(gpsum):
        for layer in range(nl):
            d = subD[i, layer]
            GHF_k[i, layer] = 0.138 - 1.01e-3 * d + 3.233e-6 * d**2

    GHF_C = np.empty(gpsum)
    hcap_sub = np.empty(gpsum)
    for i in range(gpsum):
        k0, k1 = GHF_k[i, 0], GHF_k[i, 1]
        z0, z1 = subZ[i, 0], subZ[i, 1]
        GHF_C[i] = (k0 * z0 + 0.5 * k1 * z1) / (z0 + 0.5 * z1) ** 2

        c0 = subD[i, 0] * (152.2 + 7.122 * subT[i, 0])
        c1 = subD[i, 1] * (152.2 + 7.122 * subT[i, 1])
        hcap_sub[i] = c0 * z0 + 0.5 * c1 * z1

    return GHF_k, GHF_C, hcap_sub


class TestConductance(unittest.TestCase):
    def test_matches_pointwise_reference(self):
        """Densities and thicknesses differ across both grid points and layers, and a third
        layer is included to confirm only the top two feed into GHF_C."""
        subD = np.array(
            [
                [350.0, 550.0, 900.0],
                [200.0, 800.0, 400.0],
            ]
        )
        subZ = np.array(
            [
                [0.10, 0.20, 5.0],
                [0.05, 0.40, 3.0],
            ]
        )
        subT = np.array(
            [
                [260.0, 262.0, 265.0],
                [268.0, 270.0, 272.0],
            ]
        )
        OUT = {"subD": subD, "subZ": subZ, "subT": subT}

        GHF_k, GHF_C, hcap_sub = LOOP_EBM_GHF.conductance(OUT)
        expected_k, expected_C, expected_hcap_sub = _reference_conductance(subD, subZ, subT)

        np.testing.assert_allclose(GHF_k, expected_k)
        np.testing.assert_allclose(GHF_C, expected_C)
        np.testing.assert_allclose(hcap_sub, expected_hcap_sub)

    def test_shapes(self):
        """GHF_k keeps one conductivity per layer; GHF_C and hcap_sub collapse to one value
        per grid point."""
        gpsum, nl = 4, 5
        rng = np.random.default_rng(0)
        OUT = {
            "subD": rng.uniform(100.0, 900.0, size=(gpsum, nl)),
            "subZ": rng.uniform(0.05, 0.5, size=(gpsum, nl)),
            "subT": rng.uniform(250.0, 273.0, size=(gpsum, nl)),
        }

        GHF_k, GHF_C, hcap_sub = LOOP_EBM_GHF.conductance(OUT)

        self.assertEqual(GHF_k.shape, (gpsum, nl))
        self.assertEqual(GHF_C.shape, (gpsum,))
        self.assertEqual(hcap_sub.shape, (gpsum,))


def _make_out():
    return {
        "subD": np.array([[300.0, 350.0, 400.0], [500.0, 550.0, 600.0]]),
        "subZ": np.array([[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]]),
        "subT": np.array([[260.0, 261.0, 262.0], [270.0, 271.0, 272.0]]),
    }


class TestCacheIsValid(unittest.TestCase):
    def test_valid_when_cache_matches_fresh_computation(self):
        OUT = _make_out()
        OUT["ghf_k"], OUT["ghf_cond"], _ = LOOP_EBM_GHF.conductance(OUT)

        self.assertTrue(LOOP_EBM_GHF.cache_is_valid(OUT))

    def test_invalid_when_subsurface_state_changes_after_caching(self):
        OUT = _make_out()
        OUT["ghf_k"], OUT["ghf_cond"], _ = LOOP_EBM_GHF.conductance(OUT)

        # Simulate LOOP_SNOW.main updating the firn column without the cache being refreshed.
        OUT["subD"] = OUT["subD"] + 50.0

        self.assertFalse(LOOP_EBM_GHF.cache_is_valid(OUT))


class TestMainRejectsStaleCache(unittest.TestCase):
    def test_main_runs_with_valid_cache(self):
        OUT = _make_out()
        OUT["ghf_k"], OUT["ghf_cond"], _ = LOOP_EBM_GHF.conductance(OUT)
        cond = np.ones(2, dtype=bool)

        GHF = LOOP_EBM_GHF.main(OUT["subT"][:, 1] - 1.0, OUT, cond, OUT["ghf_k"], OUT["ghf_cond"])

        self.assertEqual(GHF.shape, (2,))

    def test_main_raises_on_stale_cache(self):
        OUT = _make_out()
        OUT["ghf_k"], OUT["ghf_cond"], _ = LOOP_EBM_GHF.conductance(OUT)
        cond = np.ones(2, dtype=bool)

        # Simulate LOOP_SNOW.main updating the firn column without the cache being refreshed.
        OUT["subD"] = OUT["subD"] + 50.0

        with self.assertRaises(AssertionError):
            LOOP_EBM_GHF.main(OUT["subT"][:, 1] - 1.0, OUT, cond, OUT["ghf_k"], OUT["ghf_cond"])


if __name__ == "__main__":
    unittest.main()
