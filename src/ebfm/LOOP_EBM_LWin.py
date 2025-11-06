# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause


def main(C, IN):
    """
    Computes incoming longwave radiation (LWin) based on atmospheric emissivity.
    SOURCE: Konzelmann et al. (1994)

    Parameters:
        C (dict): Constants including coefficients for emissivity and Stefan-Boltzmann constant.
            - C["b"]: Coefficient for clear-sky emissivity.
            - C["p"]: Emissivity exponent for cloud fraction.
            - C["ecl"]: Cloud emissivity coefficient.
            - C["boltz"]: Stefan-Boltzmann constant (W/m^2/K^4).
        IN (dict): Input data including vapor pressure (VP), temperature (T), and cloud cover (C).
            - IN["VP"]: Vapor pressure (Pa).
            - IN["T"]: Air temperature (K).
            - IN["C"]: Cloud cover (fraction, 0 to 1).

    Returns:
        numpy.ndarray: Incoming longwave radiation (LWin, W/m^2).
    """

    ###########################################################
    # Atmospheric Emissivity
    ###########################################################

    # Clear-sky emissivity
    ecs = 0.23 + C["b"] * (IN["VP"] / IN["T"]) ** 0.125  # SOURCE: Konzelmann et al. (1994)

    # Sky emissivity
    e = ecs * (1.0 - IN["C"] ** C["p"]) + C["ecl"] * IN["C"] ** C["p"]

    ###########################################################
    # Incoming Longwave Radiation
    ###########################################################

    LWin = e * C["boltz"] * IN["T"] ** 4  # Stefan-Boltzmann law

    return LWin
