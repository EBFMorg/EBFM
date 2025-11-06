# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause


def main(C, Tsurf):
    """
    Calculates the outgoing longwave radiation (LWout) based on blackbody
    emission of thermal radiation.

    Parameters:
        C (dict): Constants needed for the calculations:
                  - boltz: Stefan-Boltzmann constant (W/m²/K⁴).
        Tsurf (numpy.ndarray): Surface temperature (K).

    Returns:
        numpy.ndarray: Outgoing longwave radiation (W/m²).
    """
    ###########################################################
    # Blackbody Emission of Thermal Radiation
    ###########################################################
    LWout = C["boltz"] * Tsurf**4
    return LWout
