# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from . import (
    LOOP_EBM_SHF,
    LOOP_EBM_GHF,
    LOOP_EBM_LHF,
    LOOP_EBM_LWin,
    LOOP_EBM_LWout,
    LOOP_EBM_SWin,
)
from ebfm.core import LOOP_EBM_SWout, LOOP_EBM_insolation

from ebfm.coupling import Coupler

from ebfm.core import logging

logger = logging.getLogger(__name__)


def melt_and_moisture_fluxes(C, time2, OUT) -> dict:
    """
    Melt and moisture fluxes following from the surface temperature and the heat fluxes main solved for.

    Kept out of main so that the caller decides where the results go: into OUT, or into
    OUT["ebm_diagnostics"] when another model has solved the surface energy balance for this time
    step instead (see ebfm.coupling.components.icon_land).

    Parameters:
        C (dict): Model constants and parameters.
        time2 (dict): Time-related parameters and variables.
        OUT (dict): Output variables, incl. the surface temperature and heat fluxes stored by main.

    Returns:
        dict: Surface temperature, melt energy, melt and the moisture fluxes.
    """
    Tsurf = OUT["Tsurf"]
    LHF = OUT["LHF"]

    ###########################################################
    # SURFACE MELT
    ###########################################################

    Emelt = OUT["SWin"] - OUT["SWout"] + OUT["LWin"] - OUT["LWout"] + OUT["SHF"] + LHF + OUT["GHF"]
    Emelt[Tsurf < C["T0"]] = 0.0

    melt = C["dayseconds"] * time2["dt"] * Emelt / C["Lm"] / 1e3

    ###########################################################
    # MOISTURE FLUXES
    ###########################################################

    moist_deposition = C["dayseconds"] * time2["dt"] * LHF / C["Ls"] / 1e3 * (Tsurf < C["T0"]) * (LHF > 0)
    moist_condensation = C["dayseconds"] * time2["dt"] * LHF / C["Lv"] / 1e3 * (Tsurf >= C["T0"]) * (LHF > 0)
    moist_sublimation = -C["dayseconds"] * time2["dt"] * LHF / C["Ls"] / 1e3 * (Tsurf < C["T0"]) * (LHF < 0)
    moist_evaporation = -C["dayseconds"] * time2["dt"] * LHF / C["Lv"] / 1e3 * (Tsurf >= C["T0"]) * (LHF < 0)

    ###########################################################
    # AVOID EVAPORATION OF ABSENT MELT
    ###########################################################

    moist_evaporation = np.minimum(moist_evaporation, melt)

    return {
        "Tsurf": Tsurf,
        "Emelt": Emelt,
        "melt": melt,
        "moist_deposition": moist_deposition,
        "moist_condensation": moist_condensation,
        "moist_sublimation": moist_sublimation,
        "moist_evaporation": moist_evaporation,
    }


def main(C, OUT, IN, time2, grid, cpl: Coupler) -> dict:
    """
    Surface Energy Balance Model: Solves the surface temperature and calculates the heat fluxes.

    The melt and moisture fluxes that follow from them are not part of this: they are computed by
    melt_and_moisture_fluxes, which the caller calls afterwards.

    Parameters:
        C (dict): Model constants and parameters.
        OUT (dict): Output variables to store results.
        IN (dict): Input data for the model.
        time (dict): Time-related parameters and variables.
        grid (dict): Model grid information.
        cpl (Coupler): Coupling object for data exchange with external models.

    Returns:
        dict: Updated OUT dictionary containing the surface temperature and the heat fluxes.
    """
    logger.debug("Starting LOOP_EBM...")
    ###########################################################
    # SOLVE THE SURFACE ENERGY BALANCE
    ###########################################################

    # Compute SWin, SWout, LWin (independent of surface temperature); GHF conductance comes
    # from OUT's per-time-step cache.
    OUT = LOOP_EBM_insolation.main(grid, time2, OUT)
    SWin, OUT = LOOP_EBM_SWin.main(C, OUT, IN, grid, cpl)

    # TODO: better do this before calling LOOP_EBM.main
    if cpl.has_coupling_to("icon_atmo"):
        LWin = IN["LWin"]
    else:
        LWin = LOOP_EBM_LWin.main(C, IN)

    SWout, OUT = LOOP_EBM_SWout.main(C, time2, OUT, SWin)
    GHF_k, GHF_C = OUT["ghf_k"], OUT["ghf_cond"]

    # Precompute reusable constant arrays
    gpsum = OUT["subT"].shape[0]
    condition_mask = np.ones(gpsum, dtype=bool)

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
    # HEAT FLUXES AT THE SURFACE TEMPERATURE
    ###########################################################

    # Ensure surface temperature does not exceed the melting point
    Tmid[np.abs(Tmid - C["T0"]) < C["dTacc"]] -= C["dTacc"]
    Tmid = np.minimum(Tmid, C["T0"])

    LWout = LOOP_EBM_LWout.main(C, Tmid)
    LHF = LOOP_EBM_LHF.main(C, Tmid, IN, condition_mask)
    SHF = LOOP_EBM_SHF.main(C, Tmid, IN, condition_mask)
    GHF = LOOP_EBM_GHF.main(Tmid, OUT, condition_mask, GHF_k, GHF_C)

    ###########################################################
    # STORE RELEVANT VARIABLES IN OUT
    ###########################################################

    OUT["Tsurf"] = Tmid
    OUT["LHF"] = LHF
    OUT["SWin"] = SWin
    OUT["SWout"] = SWout
    OUT["LWin"] = LWin
    OUT["LWout"] = LWout
    OUT["SHF"] = SHF
    OUT["GHF"] = GHF

    return OUT
