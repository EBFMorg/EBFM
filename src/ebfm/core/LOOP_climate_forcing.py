# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm.coupling import Coupler

from datetime import datetime, timedelta

from .LOOP_general_functions import is_first_time_step


def main(C, grid, IN, t, time, OUT, cpl: Coupler) -> tuple[dict, dict]:
    """
    Meteorological forcing: Specify or read meteorological input and derive
    associated meteorological fields.

    Parameters:
        C (dict): Constants for the model.
        grid (dict): Grid information.
        IN (dict): Meteorological input variables.
        t (int): Current time step.
        time (dict): Time-related variables.
        OUT (dict): Output variables from the model.
        cpl (Coupler): Coupling object for data exchange with external models.

    Returns:
        Updated IN and OUT dictionaries. Fields include:
        - IN['T']: Air temperature (K)
        - IN['P']: Precipitation (m w.e.)
        - IN['C']: Cloud cover (fraction)
        - IN['RH']: Relative humidity (fraction)
        - IN['WS']: Wind speed (m s-1)
        - IN['Pres']: Air pressure (Pa)
        OUT contains a copy of these fields (required in
        LOOP_write_to_file)
    """
    ###########################################################
    # SPECIFY/READ METEO FORCING
    ###########################################################
    if not cpl.has_coupling_to("icon_atmo"):
        IN = set_random_weather_data(IN, C, time, grid)

    ###########################################################
    # DERIVED METEOROLOGICAL FIELDS
    ###########################################################

    # Annual snow accumulation
    OUT["ys"] = (1.0 - (1.0 / (C["yeardays"] / time["dt"]))) * OUT["ys"] + IN["P"] * 1e3
    logys = np.log(OUT["ys"])
    IN["yearsnow"] = np.tile(OUT["ys"][:, np.newaxis], (1, grid["nl"]))
    IN["logyearsnow"] = np.tile(logys[:, np.newaxis], (1, grid["nl"]))

    # Vapor pressure, relative and specific humidity
    VPsat = C["VP0"] * np.exp(C["Lv"] / C["Rv"] * (1.0 / 273.15 - 1.0 / IN["T"])) * (IN["T"] >= 273.15) + C[
        "VP0"
    ] * np.exp(C["Ls"] / C["Rv"] * (1.0 / 273.15 - 1.0 / IN["T"])) * (IN["T"] < 273.15)

    if cpl.has_coupling_to("icon_atmo"):  # q from ICON, calculate VP and RH
        IN["VP"] = IN["q"] * IN["Pres"] / C["eps"]
        IN["RH"] = IN["VP"] / VPsat
    else:  # RH from input, calculate VP and q
        IN["VP"] = IN["RH"] * VPsat
        IN["q"] = IN["RH"] * (VPsat * C["eps"] / IN["Pres"])

    # Air density
    IN["Dair"] = IN["Pres"] / (C["Rd"] * IN["T"])

    # Time since last snowfall event
    snowfall_mask = (IN["snow"] / (time["dt"] * C["dayseconds"])) > C["Pthres"]
    OUT["timelastsnow"][snowfall_mask] = time["TCUR"]
    if is_first_time_step(t):
        OUT["timelastsnow"][:] = time["TCUR"]

    # Potential temperature and lapse rate
    IN["Theta"] = IN["T"] * (C["Pref"] / IN["Pres"]) ** (C["Rd"] / C["Cp"])
    all_same = np.all(grid["z"] == grid["z"][0])
    if not all_same:
        poly_coeff = np.polyfit(grid["z"], IN["Theta"], deg=1)
        IN["Theta_lapse"] = max(poly_coeff[0], 0.0015)
    else:
        IN["Theta_lapse"] = 0.0015

    ###########################################################
    # STORE RELEVANT VARIABLES IN OUT
    ###########################################################
    OUT["climT"] = IN["T"]
    OUT["climP"] = IN["P"]
    OUT["climC"] = IN["C"]
    OUT["climRH"] = IN["RH"]
    OUT["climWS"] = IN["WS"]
    OUT["climPres"] = IN["Pres"]
    OUT["climsnow"] = IN["snow"]
    OUT["climrain"] = IN["rain"]

    return IN, OUT


def set_random_weather_data(IN, C, time, grid):
    """
    Specify or read meteorological data for the current time-step.

    Parameters:
        IN (dict): Meteorological input variables.
        C (dict): Constants for the model.
        time (dict): Time-related variables.
        grid (dict): Grid information.

    Returns:
        dict: Updated IN dictionary with meteorological data.
    """
    ##############################
    # Example: Random Conditions
    ##############################
    yearfrac = time["TCUR"].timetuple().tm_yday / C["yeardays"]

    # Air temperature (K)
    T_amplitude = 10.0  # Seasonal temperature amplitude (K)
    T_mean_sea_level = 268.0  # Mean sea level temperature (K)
    T_lapse_rate = -0.005  # Temperature lapse rate (K m-1)
    IN["T"] = T_mean_sea_level + T_amplitude * np.sin(2 * np.pi * yearfrac - 0.65 * np.pi)
    IN["T"] += T_lapse_rate * grid["z"]

    # Precipitation (m w.e.)
    P_annual_sea_level = 0.5  # Annual precipitation at sea level (m w.e.)
    P_z_gradient = 0.1  # Precipitation - elevation gradient (% m-1)
    t_prev: datetime = time["TCUR"] - timedelta(days=time["dt"])
    day_of_week_prev_step = t_prev.isoweekday()
    day_of_week = time["TCUR"].isoweekday()
    # trigger precitipation event once every day of week "1"
    if (day_of_week == 1) and (day_of_week != day_of_week_prev_step):
        IN["P"][:] = (P_annual_sea_level / 52.0) * (1 + P_z_gradient * grid["z"] / 100.0)
    else:
        IN["P"][:] = 0.0

    # Cloud cover (fraction)
    IN["C"][:] = 1.0 if time["TCUR"].isocalendar()[1] % 2 == 0 else 0.0

    # Relative humidity (fraction)
    IN["RH"][:] = 0.8 if time["TCUR"].isocalendar()[1] % 2 == 0 else 0.5

    # Wind speed (m s-1)
    max_WS = 10.0  # Max wind speed
    IN["WS"][:] = np.random.uniform(0.0, max_WS, size=grid["gpsum"])

    # Air pressure (Pa)
    Pres_sea_level = 1015e2  # Sea level pressure (Pa)
    IN["Pres"][:] = Pres_sea_level * np.exp(-1.244e-4 * grid["z"])

    # Snowfall and rainfall
    IN["snow"] = IN["P"] * (IN["T"] < C["rainsnowT"] - 1)
    IN["rain"] = IN["P"] * (IN["T"] > C["rainsnowT"] + 1)
    in_between_mask = (IN["T"] < C["rainsnowT"] + 1) & (IN["T"] > C["rainsnowT"] - 1)
    IN["snow"] += IN["P"] * (C["rainsnowT"] - IN["T"] + 1) / 2 * in_between_mask
    IN["rain"] += IN["P"] * (1 + IN["T"] - C["rainsnowT"]) / 2 * in_between_mask

    return IN
