# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def main(C, Tsurf, IN, cond):
    """
    Calculates turbulent latent heat flux (LHF) based on bulk equations
    and turbulent exchange coefficients.

    Parameters:
        C (dict): Constants needed for the calculations:
                  - k_turb: Turbulence constant
                  - g: Acceleration due to gravity (m/s^2)
                  - T0: Reference temperature (K)
                  - Theta_lapse: Temperature lapse rate
                  - Pr: Prandtl number
                  - turb: Background turbulent exchange coefficient
                  - VP0: Reference vapor pressure (Pa)
                  - Ls: Latent heat of sublimation (J/kg)
                  - Lv: Latent heat of vaporization (J/kg)
                  - Rv: Gas constant for water vapor (J/kg/K)
                  - eps: Ratio of molecular weights (dimensionless)
                  - dTacc: Accuracy for convergence
        Tsurf (numpy.ndarray): Surface temperature (K).
        IN (dict): Input parameters, containing:
                   - T: Air temperature (K)
                   - Dair: Air diffusion coefficient (m^2/s)
                   - VP: Vapor pressure in the air (Pa)
                   - Pres: Atmospheric pressure (Pa)
        cond (numpy.ndarray): Condition mask (boolean array for lat/lon points to process).

    Returns:
        numpy.ndarray: Latent heat flux (LHF) values for the grid points.
    """

    ###########################################################
    # Turbulent Exchange Coefficient
    ###########################################################
    C_kat = np.maximum(
        C["k_turb"] * (IN["T"][cond] - Tsurf) * np.sqrt(C["g"] / (C["T0"] * (IN["Theta_lapse"] * C["Pr"]))), 0
    )
    C_turb = 0.5 * (C["turb"] + C_kat)

    ###########################################################
    # Saturation Vapor Pressure (Clausius-Clapeyron)
    ###########################################################
    VPsurf = C["VP0"] * np.exp(C["Ls"] / C["Rv"] / 273.15 * (1 - 273.15 / Tsurf)) * (Tsurf < 273.15) + C[
        "VP0"
    ] * np.exp(C["Lv"] / C["Rv"] / 273.15 * (1 - 273.15 / Tsurf)) * (Tsurf >= 273.15)

    ###########################################################
    # Latent Heat Flux (bulk equations)
    ###########################################################
    LHF = C["eps"] * IN["Dair"][cond] * C["Ls"] * C_turb * (IN["VP"][cond] - VPsurf) / IN["Pres"][cond] * (
        Tsurf < 273.15
    ) + C["eps"] * IN["Dair"][cond] * C["Lv"] * C_turb * (IN["VP"][cond] - VPsurf) / IN["Pres"][cond] * (
        Tsurf >= 273.15
    )

    return LHF
