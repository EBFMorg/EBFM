# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm import (
    LOOP_EBM_SHF,
    LOOP_EBM_GHF,
    LOOP_EBM_LHF,
    LOOP_EBM_LWin,
    LOOP_EBM_LWout,
    LOOP_EBM_SWin,
)
from ebfm import LOOP_EBM_SWout, LOOP_EBM_insolation

from couplers import Coupler


def main(C, OUT, IN, time2, grid, cpl: Coupler) -> dict:
    """
    Surface Energy Balance Model: Calculates heat fluxes, surface temperature, melt rates, and moisture fluxes.

    Parameters:
        C (dict): Model constants and parameters.
        OUT (dict): Output variables to store results.
        IN (dict): Input data for the model.
        time (dict): Time-related parameters and variables.
        grid (dict): Model grid information.
        cpl (Coupler): Coupling object for data exchange with external models.

    Returns:
        dict: Updated OUT dictionary containing energy balance results.
    """
    ###########################################################
    # SOLVE THE SURFACE ENERGY BALANCE
    ###########################################################

    # Compute SWin, SWout, LWin and GHF (independent of surface temperature)
    OUT = LOOP_EBM_insolation.main(grid, time2, OUT)
    SWin, OUT = LOOP_EBM_SWin.main(C, OUT, IN, grid, cpl)

    # TODO: better do this before calling LOOP_EBM.main
    if cpl.couple_to_icon_atmo:
        LWin = IN["LWin"]
    else:
        LWin = LOOP_EBM_LWin.main(C, IN)

    SWout, OUT = LOOP_EBM_SWout.main(C, time2, OUT, SWin)
    GHF_k = 0.138 - 1.01e-3 * OUT["subD"] + 3.233e-6 * OUT["subD"] ** 2
    GHF_C = (GHF_k[:, 0] * OUT["subZ"][:, 0] + 0.5 * GHF_k[:, 1] * OUT["subZ"][:, 1]) / (
        OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1]
    ) ** 2

    # Precompute reusable constant arrays
    condition_mask = np.ones(grid["gpsum"], dtype=bool)

    # Set initial temperature range
    Tlow = OUT["Tsurf"] - 40.0
    Thigh = OUT["Tsurf"] + 40.0
    dT = Thigh - Tlow

    for c in range(20):
        # Compute midpoint and half-step size
        Tmid = (Tlow + Thigh) / 2.0
        dT *= 0.5

        # Surface energy balance at Tlow
        ebal_Tlow = (
            SWin
            - SWout
            + LWin
            - LOOP_EBM_LWout.main(C, Tlow)
            + LOOP_EBM_LHF.main(C, Tlow, IN, condition_mask)
            + LOOP_EBM_SHF.main(C, Tlow, IN, condition_mask)
            + LOOP_EBM_GHF.main(Tlow, OUT, condition_mask, GHF_k, GHF_C)
        )

        # Surface energy balance at Tmid
        ebal_Tmid = (
            SWin
            - SWout
            + LWin
            - LOOP_EBM_LWout.main(C, Tmid)
            + LOOP_EBM_LHF.main(C, Tmid, IN, condition_mask)
            + LOOP_EBM_SHF.main(C, Tmid, IN, condition_mask)
            + LOOP_EBM_GHF.main(Tmid, OUT, condition_mask, GHF_k, GHF_C)
        )

        # Update temperature range based on energy balance sign
        cond_EB = ebal_Tmid * ebal_Tlow < 0
        Thigh[cond_EB] = Tmid[cond_EB]
        Tlow[~cond_EB] = Tmid[~cond_EB]

        # Stop if temperature change is below the predefined limit
        if np.max(dT) < C["dTacc"]:
            break

        if c == 19:
            raise ValueError("Energy balance did not converge below limit C.dTacc")

    ###########################################################
    # SURFACE MELT
    ###########################################################

    # Ensure surface temperature does not exceed the melting point
    Tmid[np.abs(Tmid - C["T0"]) < C["dTacc"]] -= C["dTacc"]
    Tmid = np.minimum(Tmid, C["T0"])

    LWout = LOOP_EBM_LWout.main(C, Tmid)
    LHF = LOOP_EBM_LHF.main(C, Tmid, IN, condition_mask)
    SHF = LOOP_EBM_SHF.main(C, Tmid, IN, condition_mask)
    GHF = LOOP_EBM_GHF.main(Tmid, OUT, condition_mask, GHF_k, GHF_C)

    Emelt = SWin - SWout + LWin - LWout + SHF + LHF + GHF
    Emelt[Tmid < C["T0"]] = 0.0

    OUT["melt"] = C["dayseconds"] * time2["dt"] * Emelt / C["Lm"] / 1e3

    ###########################################################
    # MOISTURE FLUXES
    ###########################################################

    OUT["moist_deposition"] = C["dayseconds"] * time2["dt"] * LHF / C["Ls"] / 1e3 * (Tmid < C["T0"]) * (LHF > 0)
    OUT["moist_condensation"] = C["dayseconds"] * time2["dt"] * LHF / C["Lv"] / 1e3 * (Tmid >= C["T0"]) * (LHF > 0)
    OUT["moist_sublimation"] = -C["dayseconds"] * time2["dt"] * LHF / C["Ls"] / 1e3 * (Tmid < C["T0"]) * (LHF < 0)
    OUT["moist_evaporation"] = -C["dayseconds"] * time2["dt"] * LHF / C["Lv"] / 1e3 * (Tmid >= C["T0"]) * (LHF < 0)

    ###########################################################
    # AVOID EVAPORATION OF ABSENT MELT
    ###########################################################

    max_evap = OUT["melt"]
    OUT["moist_evaporation"] = np.minimum(OUT["moist_evaporation"], max_evap)

    ###########################################################
    # STORE RELEVANT VARIABLES IN OUT
    ###########################################################

    OUT["Tsurf"] = Tmid
    OUT["Emelt"] = Emelt
    OUT["LHF"] = LHF
    OUT["SWin"] = SWin
    OUT["SWout"] = SWout
    OUT["LWin"] = LWin
    OUT["LWout"] = LWout
    OUT["SHF"] = SHF
    OUT["GHF"] = GHF

    return OUT
