# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm.constants.materials import Ice


def main(OUT, IN):
    """
    Update the climatic mass balance and snow mass.

    Parameters:
    - OUT: Dictionary containing output variables.
    - IN: Dictionary containing input climatic variables.

    Returns:
    - Updated `OUT` dictionary.
    """

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
    OUT["snowmass"][np.all(OUT["subD"] >= Ice.DENSITY, axis=1)] = 0

    return OUT
