# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Surface energy balance provided by ICON-Land (JSBACH).

When EBFM is coupled to ICON-Land, the surface energy balance of the glacier tile is computed
by JSBACH at every ICON time step and EBFM receives its results averaged over the EBFM time
step (see ebfm.coupling.components.icon_land): surface temperature, melt and
evapotranspiration. These drive EBFM's snow/firn model (LOOP_SNOW) and the mass balance.
EBFM's own energy balance (LOOP_EBM) is still evaluated every time step from the atmospheric
forcing and the current firn state, but only as a diagnostic: its results are kept under
OUT["ebm_*"] so that the two energy balances can be compared.
"""

import numpy as np

from ebfm.core import logging

logger = logging.getLogger(__name__)

# Fields that must have been received from ICON-Land to replace EBFM's energy balance
REQUIRED_FIELDS = ("lice_t_srf", "lice_melt", "lice_evapotrans")

# Results of EBFM's own energy balance that are kept as diagnostics (OUT["ebm_<name>"])
EBM_DIAGNOSTICS = (
    "Tsurf",
    "melt",
    "Emelt",
    "moist_sublimation",
    "moist_evaporation",
    "moist_deposition",
    "moist_condensation",
)


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
            "using EBFM's own surface energy balance for this time step."
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


def main(C: dict, OUT: dict, IN: dict, time2: dict) -> dict:
    """
    Replace the results of EBFM's energy balance in OUT by the fields received from ICON-Land.

    Must be called after LOOP_EBM has computed EBFM's own energy balance: those results are
    kept as OUT["ebm_*"] and Tsurf, melt, Emelt and the moist_* terms are set from ICON-Land.
    The radiative and turbulent flux diagnostics (SWin, SWout, LWin, LWout, SHF, LHF, GHF) remain
    those of EBFM's own balance; JSBACH's fluxes are available in the ICON-Land output.

    @param[in] C model constants
    @param[in,out] OUT output variables
    @param[in] IN input variables incl. the received IN["lice_*"] fields
    @param[in] time2 time information (dt in days)

    @returns updated OUT
    """
    logger.debug("Using the surface energy balance received from ICON-Land...")

    # Keep EBFM's own results as diagnostics
    for name in EBM_DIAGNOSTICS:
        OUT[f"ebm_{name}"] = OUT[name]

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

    logger.debug(
        "Surface energy balance ICON-Land vs EBFM (mean over grid): "
        f"Tsurf {np.mean(Tsurf):.2f} vs {np.mean(OUT['ebm_Tsurf']):.2f} K, "
        f"melt {np.mean(melt):.3e} vs {np.mean(OUT['ebm_melt']):.3e} m w.e., "
        f"sublimation {np.mean(moist['moist_sublimation']):.3e} vs {np.mean(OUT['ebm_moist_sublimation']):.3e} m w.e."
    )

    return OUT
