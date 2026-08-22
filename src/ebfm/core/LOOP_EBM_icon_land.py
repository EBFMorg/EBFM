# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Surface energy balance provided by ICON-Land (JSBACH).

When EBFM is coupled to ICON-Land, the surface energy balance of the glacier tile is computed
by JSBACH at every ICON time step and EBFM receives its results averaged over the EBFM time
step (see ebfm.coupling.components.icon_land): surface temperature, melt, evapotranspiration
and the surface energy fluxes. In that case EBFM does not solve its own energy balance but
derives the quantities needed by the snow/firn model (LOOP_SNOW) and the mass balance from
the received fields.
"""

import numpy as np

from ebfm.core import LOOP_EBM_LWout
from ebfm.core import logging

logger = logging.getLogger(__name__)

# Fields that must have been received from ICON-Land to replace EBFM's energy balance
REQUIRED_FIELDS = ("lice_t_srf", "lice_melt", "lice_evapotrans")
# Diagnostic fluxes (W m-2, positive into the surface, as in EBFM) that are stored if received
DIAGNOSTIC_FLUXES = {"SHF": "lice_hfss", "LHF": "lice_hfls", "GHF": "lice_ghf"}


def is_available(IN: dict, cpl) -> bool:
    """
    Whether the surface energy balance of ICON-Land is to be used in this time step.

    @param[in] IN input dictionary (received fields are stored as IN["lice_*"])
    @param[in] cpl coupler

    @returns True if EBFM is coupled to ICON-Land and all required fields have been received
    """
    if not cpl.has_coupling_to("icon_land"):
        return False
    missing = [name for name in REQUIRED_FIELDS if name not in IN]
    if missing:
        logger.warning(
            f"Coupled to ICON-Land, but the fields {missing} have not been received: "
            "falling back to EBFM's own surface energy balance for this time step."
        )
        return False
    return True


def partition_evapotrans(evapotrans: np.ndarray, Tsurf: np.ndarray, melt: np.ndarray, T0: float) -> dict:
    """
    Partition the evapotranspiration received from ICON-Land into EBFM's moisture terms.

    Follows the logic of LOOP_EBM: for a surface below the melting point the latent heat
    exchange is sublimation (loss) or deposition (gain), at the melting point evaporation or
    condensation. Evaporation is limited by the available melt water as in LOOP_EBM.

    @param[in] evapotrans evapotranspiration incl. sublimation in m w.e. per time step, negative upward (loss)
    @param[in] Tsurf surface temperature (K)
    @param[in] melt melt in m w.e. per time step
    @param[in] T0 melting point (K)

    @returns dictionary with moist_sublimation, moist_evaporation (losses, positive),
             moist_deposition, moist_condensation (gains, positive), all in m w.e. per time step
    """
    frozen = Tsurf < T0
    loss = np.maximum(-evapotrans, 0.0)
    gain = np.maximum(evapotrans, 0.0)

    moist_sublimation = np.where(frozen, loss, 0.0)
    moist_evaporation = np.where(~frozen, loss, 0.0)
    moist_evaporation = np.minimum(moist_evaporation, melt)
    moist_deposition = np.where(frozen, gain, 0.0)
    moist_condensation = np.where(~frozen, gain, 0.0)

    return {
        "moist_sublimation": moist_sublimation,
        "moist_evaporation": moist_evaporation,
        "moist_deposition": moist_deposition,
        "moist_condensation": moist_condensation,
    }


def main(C: dict, OUT: dict, IN: dict, time2: dict, SWin: np.ndarray, SWout: np.ndarray, LWin: np.ndarray) -> dict:
    """
    Fill the energy balance results in OUT from the fields received from ICON-Land.

    Replaces the iterative solution of the energy balance in LOOP_EBM; the shortwave terms
    (incl. the albedo evolution) are still computed by EBFM and passed in.

    @param[in] C model constants
    @param[in,out] OUT output variables (Tsurf, melt, moist_*, Emelt and the fluxes are set)
    @param[in] IN input variables incl. the received IN["lice_*"] fields
    @param[in] time2 time information (dt in days)
    @param[in] SWin, SWout, LWin shortwave and incoming longwave radiation (W m-2)

    @returns updated OUT
    """
    logger.debug("Using the surface energy balance received from ICON-Land...")

    # Surface temperature: JSBACH limits the glacier surface temperature to the melting point;
    # the time average may not exceed it either, but guard against round-off.
    Tsurf = np.minimum(np.asarray(IN["lice_t_srf"], dtype=float), C["T0"])

    # Melt (m w.e. per time step, already converted by the coupling component)
    melt = np.maximum(np.asarray(IN["lice_melt"], dtype=float), 0.0)

    # Moisture terms from the evapotranspiration (m w.e. per time step, negative upward)
    moist = partition_evapotrans(np.asarray(IN["lice_evapotrans"], dtype=float), Tsurf, melt, C["T0"])

    OUT["Tsurf"] = Tsurf
    OUT["melt"] = melt
    OUT.update(moist)

    # Energy equivalent of the melt (W m-2), for output only
    seconds_per_timestep = C["dayseconds"] * time2["dt"]
    OUT["Emelt"] = melt * 1e3 * C["Lm"] / seconds_per_timestep

    # Radiation terms: shortwave from EBFM, outgoing longwave from the received surface temperature
    OUT["SWin"] = SWin
    OUT["SWout"] = SWout
    OUT["LWin"] = LWin
    OUT["LWout"] = LOOP_EBM_LWout.main(C, Tsurf)

    # Turbulent and ground heat fluxes as computed by ICON-Land (diagnostics)
    for out_name, in_name in DIAGNOSTIC_FLUXES.items():
        OUT[out_name] = np.asarray(IN[in_name], dtype=float) if in_name in IN else np.zeros_like(Tsurf)

    return OUT
