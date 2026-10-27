# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import datetime

from .grid import ShadingMethod

from .config import DEFAULT_TZ


def main(grid, time2, OUT):
    """
    Calculates the unattenuated incoming solar radiation and shading due to topography.

    Parameters:
        grid (dict): Contains grid-related variables like elevation, latitude, longitude, etc.
        time2 (dict): Time data including the current time step "TCUR".
        OUT (dict): Output dictionary to store computed values.

    Returns:
        dict: Updated OUT dictionary with calculated solar radiation and shading variables.
    """

    ###########################################################
    # Top-of-atmosphere radiation, solar declination and hour angle
    ###########################################################

    # Time as a fraction of a year (in radians)
    start_of_year = datetime.datetime(time2["TCUR"].year, 1, 1, tzinfo=DEFAULT_TZ)  # January 1 of TCUR's year
    day_of_year = (time2["TCUR"] - start_of_year).days + (
        time2["TCUR"] - start_of_year
    ).seconds / 86400  # Fractional days
    trad = 2 * np.pi * day_of_year / 365.242

    # Top-of-atmosphere radiation on a surface normal to incident beam
    OUT["I0"] = 1353.0 * (1.0 + 0.034 * np.cos(trad))  # SOURCE: Meyers and Dale (1983)

    # Solar declination (d) using an approximation formula
    d = (
        0.322003
        - 22.971 * np.cos(trad)
        - 0.357898 * np.cos(2.0 * trad)
        - 0.14398 * np.cos(3.0 * trad)
        + 3.94638 * np.sin(trad)
        + 0.019334 * np.sin(2.0 * trad)
        + 0.05928 * np.sin(3.0 * trad)
    )  # SOURCE: Iqbal (1983)

    # Solar hour angle (h)
    B = 360 / 365 * (day_of_year - 81)
    Tcor_ecc = (
        9.87 * np.sin(2.0 * np.radians(B)) - 7.53 * np.cos(np.radians(B)) - 1.5 * np.sin(np.radians(B))
    )  # Correction for eccentricity

    Tcor_lon = 4 * grid["lon"]  # Correction for longitude within time-zone
    Tcor = Tcor_ecc + Tcor_lon
    LST = time2["TCUR"].hour + time2["TCUR"].minute / 60 + Tcor / 60  # Local Solar Time
    h = 15 * (LST - 12)

    lat_rad = np.radians(grid["lat"])
    d_rad = np.radians(d)
    h_rad = np.radians(h)

    OUT["lat_rad"] = lat_rad
    OUT["d_rad"] = d_rad
    OUT["h_rad"] = h_rad

    ###########################################################
    # Shading by the surrounding topography
    ###########################################################
    # Solar elevation angle (radians)
    elevationangle = np.arcsin(
        np.sin(lat_rad) * np.sin(d_rad) + np.cos(lat_rad) * np.cos(d_rad) * np.cos(h_rad)
    )  # SOURCE: Iqbal (1983)

    if grid["has_shading"]:
        gpsum = OUT["subT"].shape[0]

        # Azimuth (radians)
        cos_elevation = np.cos(elevationangle)
        azimuth_argument = (
            np.cos(h_rad) * np.cos(d_rad) * np.sin(lat_rad) - np.sin(d_rad) * np.cos(lat_rad)
        ) / cos_elevation
        azimuth_argument = np.clip(azimuth_argument, -1.0, 1.0)

        azimuth = np.where(
            h < 0,
            np.arccos(azimuth_argument),
            -np.arccos(azimuth_argument),
        )

        if grid["shading_method"] == ShadingMethod.LUT:
            # shading based on comparison of solar elevation angle with grid angle from look-up table (LUT)

            # Select nearest pre-computed azimuth angle for every grid cell (1..grid.nr_az_steps)
            az_nearest = np.rint((np.pi - azimuth) / (2.0 * np.pi) * grid["nr_az_steps"]).astype(int)
            az_nearest[az_nearest == 0] = grid["nr_az_steps"]

            # Indices into lookup table (0-based for Python)
            ind_az = az_nearest - 1 + np.arange(gpsum, dtype=int) * grid["nr_az_steps"]

            # Shading happens when the solar elevation angle <= maximum grid angle
            temp = grid["maxgridangle"]
            OUT["is_shaded"] = elevationangle <= temp.flat[ind_az]

            # For now: assume flat surface and no shading by surrounding terrain (no longer needed!)
            # OUT["is_shaded"] = np.zeros_like(grid["x"])
            # OUT["is_shaded"][elevationangle < 0] = 1
        elif grid["shading_method"] == ShadingMethod.LUT_EXTERNAL:
            # Requires YAC and Elmer/Ice -> heavy dependencies
            # TODO: Shading based on lookup tables from Elmer/Ice
            raise NotImplementedError("Shading based on look-up tables from Elmer/Ice is not implemented yet.")

    else:  # fall-back values for no shading
        # TODO: Shading based on lookup tables from Elmer/Ice
        # For now: assume flat surface and no shading by surrounding terrain
        OUT["is_shaded"] = np.zeros(grid["x"].shape, dtype=bool)
        OUT["is_shaded"][elevationangle < 0] = True

    return OUT
