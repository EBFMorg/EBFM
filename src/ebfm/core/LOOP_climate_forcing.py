# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
from datetime import timezone, timedelta

from netCDF4 import Dataset
import numpy as np

from ebfm.coupling import Coupler
from ebfm.core.config import GridConfig
from ebfm.core.grid import GridInputType

from .LOOP_general_functions import is_first_time_step

from ebfm.core import logging

logger = logging.getLogger(__name__)


def main(C, grid, IN, t, time, OUT, cpl: Coupler, config: GridConfig) -> tuple[dict, dict]:
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
    logger.debug("Starting LOOP_climate_forcing...")
    ###########################################################
    # SPECIFY/READ METEO FORCING
    ###########################################################
    if not cpl.has_coupling_to("icon_atmo"):
        if config.grid_type is GridInputType.GREENLAND:
            IN = read_Greenland_data(IN, C, time, grid, config)
        else:
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

    if (
        cpl.has_coupling_to("icon_atmo") or config.grid_type is GridInputType.GREENLAND
    ):  # q from ICON/CARRA2, calculate VP and RH
        IN["VP"] = IN["q"] * IN["Pres"] / C["eps"]
        IN["RH"][:] = np.clip(IN["VP"] / VPsat, 0.0, 1.0)
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


def read_Greenland_data(IN, C, time, grid, config: GridConfig):
    """
    Read vectorized meteorological data for the current time-step from a preprocessed NetCDF file.

    Parameters:
        IN (dict): Meteorological input variables.
        C (dict): Constants for the model.
        time (dict): Time-related variables.
        grid (dict): Grid information.
        config (GridConfig): Grid configuration.

    Returns:
        dict: Updated IN dictionary with meteorological data.
    """
    forcing_dir = config.mesh_file.parent

    model_time_utc = time["TCUR"] - timedelta(hours=time["dT_UTC"])
    model_time_seconds = model_time_utc.replace(tzinfo=timezone.utc).timestamp()

    file_group = 1 if model_time_utc.month <= 6 else 2
    forcing_prefix = f"{model_time_utc.year}_{file_group}"
    forcing_file = forcing_dir / f"{forcing_prefix}_forcing_vectorized.nc"

    forcing_variables = ("C", "T", "Pres", "WS", "P", "q")

    def close_cached_datasets(cache):
        dataset = cache.get("dataset")
        if dataset is not None:
            dataset.close()

    if not hasattr(read_Greenland_data, "_cache"):
        read_Greenland_data._cache = {}

    cache = read_Greenland_data._cache

    if cache.get("forcing_file") != forcing_file:
        close_cached_datasets(cache)

        logger.info(f"Opening vectorized Greenland forcing file {forcing_file}...")

        dataset = Dataset(forcing_file, "r")
        forcing_time = np.asarray(dataset.variables["time"][:], dtype=float)

        variables = {}
        for variable_name in forcing_variables:
            if variable_name not in dataset.variables:
                dataset.close()
                raise KeyError(f"Missing `{variable_name}` in vectorized Greenland forcing file {forcing_file}.")

            variable = dataset.variables[variable_name]

            if variable.ndim != 2:
                dataset.close()
                raise ValueError(
                    f"`{variable_name}` in {forcing_file} must have dimensions (time, gpsum). "
                    f"Got shape {variable.shape}."
                )

            if variable.shape[1] != IN["C"].shape[0]:
                dataset.close()
                raise ValueError(
                    f"`{variable_name}` vector length must match model vector length. "
                    f"Got {variable.shape[1]}, model={IN['C'].shape[0]}."
                )

            variables[variable_name] = variable

        cache.clear()
        cache.update(
            {
                "forcing_file": forcing_file,
                "time": forcing_time,
                "dataset": dataset,
                "variables": variables,
                "window_start": None,
                "window_end": None,
                "data": {},
            }
        )

        logger.info(f"Done opening vectorized Greenland forcing file {forcing_file}.")

    time_index = int(np.argmin(np.abs(cache["time"] - model_time_seconds)))

    time_window_size = 16
    if cache["window_start"] is None or time_index < cache["window_start"] or time_index >= cache["window_end"]:
        window_start = time_index
        window_end = min(window_start + time_window_size, cache["time"].shape[0])

        cache["data"] = {
            variable_name: np.asarray(
                cache["variables"][variable_name][window_start:window_end, :],
                dtype=np.float32,
            )
            for variable_name in forcing_variables
        }
        cache["window_start"] = window_start
        cache["window_end"] = window_end
        logger.info(f"Read {time_window_size} time steps from vectorized Greenland forcing file.")

    local_time_index = time_index - cache["window_start"]

    def read_current_vectorized_field(variable_name: str) -> np.ndarray:
        field_vector = cache["data"][variable_name][local_time_index, :]

        if field_vector.shape != IN["C"].shape:
            raise ValueError(
                f"Vectorized `{variable_name}` must match model vector shape. "
                f"Got {variable_name}={field_vector.shape}, model={IN['C'].shape}."
            )

        return field_vector

    IN["C"][:] = np.clip(read_current_vectorized_field("C"), 0.0, 1.0)
    IN["T"][:] = read_current_vectorized_field("T")
    IN["Pres"][:] = read_current_vectorized_field("Pres")
    IN["WS"][:] = read_current_vectorized_field("WS")
    IN["P"][:] = read_current_vectorized_field("P")
    IN["q"][:] = read_current_vectorized_field("q")

    IN["snow"] = IN["P"] * (IN["T"] < C["rainsnowT"] - 1)
    IN["rain"] = IN["P"] * (IN["T"] > C["rainsnowT"] + 1)
    in_between_mask = (IN["T"] < C["rainsnowT"] + 1) & (IN["T"] > C["rainsnowT"] - 1)
    IN["snow"] += IN["P"] * (C["rainsnowT"] - IN["T"] + 1) / 2 * in_between_mask
    IN["rain"] += IN["P"] * (1 + IN["T"] - C["rainsnowT"]) / 2 * in_between_mask

    return IN


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
    T_mean_sea_level = 269.0  # Mean sea level temperature (K)
    T_lapse_rate = -0.005  # Temperature lapse rate (K m-1)
    IN["T"] = T_mean_sea_level + T_amplitude * np.sin(2 * np.pi * yearfrac - 0.65 * np.pi)
    IN["T"] += T_lapse_rate * grid["z"]

    # Precipitation (m w.e.)
    P_annual_sea_level = 0.5  # Annual precipitation at sea level (m w.e.)
    P_z_gradient = 0.1  # Precipitation - elevation gradient (% m-1)
    day_of_week = time["TCUR"].isoweekday()
    # trigger precitipation event once every day of week "1"
    if day_of_week == 1:
        IN["P"][:] = (P_annual_sea_level / (52.0 / time["dt"])) * (1 + P_z_gradient * grid["z"] / 100.0)
    else:
        IN["P"][:] = 0.0

    # Cloud cover (fraction)
    IN["C"][:] = 1.0 if time["TCUR"].isocalendar()[1] % 2 == 0 else 0.0

    # Relative humidity (fraction)
    IN["RH"][:] = 0.8 if time["TCUR"].isocalendar()[1] % 2 == 0 else 0.5

    # Wind speed (m s-1)
    max_WS = 10.0  # Max wind speed
    IN["WS"][:] = np.random.uniform(0.0, max_WS)

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
