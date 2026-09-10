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


def _reference_conductance(subD, subZ):
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
    for i in range(gpsum):
        k0, k1 = GHF_k[i, 0], GHF_k[i, 1]
        z0, z1 = subZ[i, 0], subZ[i, 1]
        GHF_C[i] = (k0 * z0 + 0.5 * k1 * z1) / (z0 + 0.5 * z1) ** 2

    return GHF_k, GHF_C


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
        OUT = {"subD": subD, "subZ": subZ}

        GHF_k, GHF_C = LOOP_EBM_GHF.conductance(OUT)
        expected_k, expected_C = _reference_conductance(subD, subZ)

        np.testing.assert_allclose(GHF_k, expected_k)
        np.testing.assert_allclose(GHF_C, expected_C)

    def test_shapes(self):
        """GHF_k keeps one conductivity per layer; GHF_C collapses to one conductance per
        grid point."""
        gpsum, nl = 4, 5
        rng = np.random.default_rng(0)
        OUT = {
            "subD": rng.uniform(100.0, 900.0, size=(gpsum, nl)),
            "subZ": rng.uniform(0.05, 0.5, size=(gpsum, nl)),
        }

        GHF_k, GHF_C = LOOP_EBM_GHF.conductance(OUT)

        self.assertEqual(GHF_k.shape, (gpsum, nl))
        self.assertEqual(GHF_C.shape, (gpsum,))


if __name__ == "__main__":
    unittest.main()
