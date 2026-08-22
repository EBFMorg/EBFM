# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause


def conductance(OUT):
    """
    Effective conductance (W m-2 K-1) between the surface and the midpoint of the second
    subsurface layer, from the density-dependent conductivity of the two top layers.

    Parameters:
        OUT (dict): A dictionary containing subD (densities, kg m-3) and subZ (layer thicknesses, m).

    Returns:
        tuple: GHF_k (conductivities of all layers, W m-1 K-1) and GHF_C (conductance, W m-2 K-1).
    """
    GHF_k = 0.138 - 1.01e-3 * OUT["subD"] + 3.233e-6 * OUT["subD"] ** 2
    GHF_C = (GHF_k[:, 0] * OUT["subZ"][:, 0] + 0.5 * GHF_k[:, 1] * OUT["subZ"][:, 1]) / (
        OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1]
    ) ** 2
    return GHF_k, GHF_C


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
