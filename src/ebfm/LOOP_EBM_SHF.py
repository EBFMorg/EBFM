import numpy as np


def main(C, Tsurf, IN, cond):
    """
    Calculates turbulent sensible heat flux (SHF) based on bulk equations
    and turbulent exchange coefficients.

    Parameters:
        C (dict): Constants needed for the calculations:
                  - k_turb: Turbulence constant
                  - g: Acceleration due to gravity (m/s^2)
                  - T0: Reference temperature (K)
                  - Theta_lapse: Temperature lapse rate
                  - Pr: Prandtl number
                  - turb: Background turbulent exchange coefficient
                  - Cp: Specific heat capacity of air at constant pressure (J/kg/K)
        Tsurf (numpy.ndarray): Surface temperature (K).
        IN (dict): Input parameters, containing:
                   - T: Air temperature (K)
                   - Dair: Air diffusion coefficient (m^2/s)
        cond (numpy.ndarray): Condition mask (boolean array for grid points to process).

    Returns:
        numpy.ndarray: Sensible heat flux (SHF) values for the grid points.
    """
    ###########################################################
    # Turbulent Exchange Coefficient
    ###########################################################
    C_kat = np.maximum(
        C["k_turb"] * (IN["T"][cond] - Tsurf) * np.sqrt(C["g"] / (C["T0"] * (IN["Theta_lapse"] * C["Pr"]))), 0
    )
    C_turb = 0.5 * (C["turb"] + C_kat)

    ###########################################################
    # Sensible Heat Flux (bulk equations)
    ###########################################################
    SHF = IN["Dair"][cond] * C["Cp"] * C_turb * (IN["T"][cond] - Tsurf)

    return SHF
