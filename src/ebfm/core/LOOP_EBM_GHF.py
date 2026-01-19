# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause


def main(Tsurf, OUT, cond, GHF_k, GHF_C):
    """
    Calculates the subsurface heat flux (GHF) based on effective conductivity
    and the temperature gradient from the surface to the midpoint of the second
    subsurface layer.

    Parameters:
        Tsurf (numpy.ndarray): Surface temperature (K).
        OUT (dict): A dictionary containing:
                    - subD (numpy.ndarray): Depths (m) of subsurface layers.
                    - subZ (numpy.ndarray): Layer thicknesses (m).
                    - subT (numpy.ndarray): Subsurface layer temperatures (K).
        cond (numpy.ndarray): Condition mask (boolean array for grid points to process).

    Returns:
        numpy.ndarray: Subsurface heat flux (GHF) for the specified points.
    """

    ###########################################################
    # Subsurface Heat Flux (bulk equation)
    ###########################################################
    GHF = GHF_C[cond] * (OUT["subT"][cond, 1] - Tsurf)

    return GHF
