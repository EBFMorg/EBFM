# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm.core import logging

logger = logging.getLogger(__name__)


def main(OUT, IN, C):
    """
    Update the climatic mass balance and snow mass.

    Parameters:
    - OUT: Dictionary containing output variables.
    - IN: Dictionary containing input climatic variables.
    - C: Dictionary containing constants like Dice.

    Returns:
    - Updated `OUT` dictionary.
    """

    logger.debug("Starting LOOP_mass_balance...")

    # Climatic mass balance
    OUT["smb"] = (
        IN["snow"]
        + IN["rain"]
        - OUT["runoff"]
        + OUT["moist_deposition"]
        + OUT["moist_condensation"]
        - OUT["moist_sublimation"]
        - OUT["moist_evaporation"]
    )

    OUT["smb_cumulative"] += OUT["smb"]

    # Snow mass
    OUT["snowmass"] = np.maximum(OUT["snowmass"] + OUT["smb"], 0)
    # On the GPU backend subD stays resident on the device across timesteps, so
    # LOOP_SNOW hands the column-wise reduction over ready-made rather than
    # having the host pull the whole density grid back for it.
    all_ice = OUT.get("all_ice_column")
    if all_ice is None:
        all_ice = np.all(OUT["subD"] >= C["Dice"], axis=1)
    OUT["snowmass"][all_ice] = 0

    return OUT
