# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm.constants.materials import FreshSnow, Firn, Ice


def main(C, time, OUT, SWin):
    """
    Computes the reflected shortwave radiation (SWout) based on surface albedo
    and incoming shortwave radiation (SWin).

    Parameters:
        C (dict): Constants and parameters, such as albedo values and t-star.
        time (dict): Contains the current time step and timestep size (dt).
        OUT (dict): Contains surface properties like temperature, snow mass, albedo, etc.
        SWin (numpy array): Incoming shortwave radiation array.

    Returns:
        tuple: SWout (reflected shortwave radiation) and updated OUT dictionary.
    """

    ###########################################################
    # Albedo Calculations
    # SOURCE: Oerlemans and Knap (1998), Bougamont et al. (2005),
    #         Van Pelt et al. (2019)
    ###########################################################

    # Conditions for ice and snow surfaces
    ice_cond = (OUT["subD"][:, 0] == Ice.DENSITY) | (OUT["snowmass"] == 0)  # Ice condition
    snow_cond = ~ice_cond  # Snow condition is the complement of ice condition

    # For a snow surface
    OUT["tstar"] = np.zeros_like(OUT["Tsurf"])  # Initialize tstar array
    OUT["tstar"][(OUT["Tsurf"] == C["T0"]) & snow_cond] = C["tstar_wet"]
    snow_below_T0_cond = (OUT["Tsurf"] < C["T0"]) & snow_cond
    OUT["tstar"][snow_below_T0_cond] = (
        C["tstar_dry"] + np.minimum(C["T0"] - OUT["Tsurf"][snow_below_T0_cond], 10) * C["tstar_K"]
    )

    alb_snow_decay_cond = snow_cond & (OUT["timelastsnow"] < time["TCUR"])
    OUT["alb_snow"][alb_snow_decay_cond] -= (
        np.maximum(OUT["alb_snow"][alb_snow_decay_cond] - Firn.ALBEDO, 0.0)
        / OUT["tstar"][alb_snow_decay_cond]
        * time["dt"]
    )

    fresh_snow_cond = (OUT["timelastsnow"] == time["TCUR"]) | ice_cond
    OUT["alb_snow"][fresh_snow_cond] = FreshSnow.ALBEDO
    OUT["albedo"][snow_cond] = OUT["alb_snow"][snow_cond]

    # For an ice surface
    OUT["albedo"][ice_cond] = Ice.ALBEDO

    ###########################################################
    # Compute Reflected Shortwave Radiation
    ###########################################################

    SWout = SWin * OUT["albedo"]

    return SWout, OUT
