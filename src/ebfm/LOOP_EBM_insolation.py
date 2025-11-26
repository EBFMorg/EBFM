# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import datetime


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
    start_of_year = datetime.datetime(time2["TCUR"].year, 1, 1)  # January 1 of TCUR's year
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
    Tcor_lon = 4 * (grid["lon"] - 15 * time2["dT_UTC"])  # Correction for longitude within time-zone
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
    elevationangle = np.arcsin(
        np.sin(lat_rad) * np.sin(d_rad) + np.cos(lat_rad) * np.cos(d_rad) * np.cos(h_rad)
    )  # SOURCE: Iqbal (1983)

    if grid["has_shading"]:
        yl = grid["Ly"]
        xl = grid["Lx"]

        # Azimuth calculation
        cos_elevation = np.cos(elevationangle)
        azimuth = np.where(
            h < 0,
            np.arccos(
                (np.cos(h_rad) * np.cos(d_rad) * np.sin(lat_rad) - np.sin(d_rad) * np.cos(lat_rad)) / cos_elevation
            ),
            -np.arccos(
                (np.cos(h_rad) * np.cos(d_rad) * np.sin(lat_rad) - np.sin(d_rad) * np.cos(lat_rad)) / cos_elevation
            ),
        )

        # Directional gradients based on azimuth
        ddx = np.zeros_like(azimuth)
        ddy = np.zeros_like(azimuth)

        ddx[azimuth <= -0.75 * np.pi] = -np.tan(np.pi + azimuth[azimuth <= -0.75 * np.pi])
        ddx[(azimuth > -0.75 * np.pi) & (azimuth <= -0.25 * np.pi)] = -1
        ddx[(azimuth > -0.25 * np.pi) & (azimuth <= 0.25 * np.pi)] = np.tan(
            azimuth[(azimuth > -0.25 * np.pi) & (azimuth <= 0.25 * np.pi)]
        )
        ddx[(azimuth > 0.25 * np.pi) & (azimuth <= 0.75 * np.pi)] = 1
        ddx[azimuth > 0.75 * np.pi] = np.tan(np.pi - azimuth[azimuth > 0.75 * np.pi])

        ddy[azimuth <= -0.75 * np.pi] = 1
        ddy[(azimuth > -0.75 * np.pi) & (azimuth <= -0.25 * np.pi)] = -np.tan(
            np.pi / 2 + azimuth[(azimuth > -0.75 * np.pi) & (azimuth <= -0.25 * np.pi)]
        )
        ddy[(azimuth > -0.25 * np.pi) & (azimuth <= 0.25 * np.pi)] = -1
        ddy[(azimuth > 0.25 * np.pi) & (azimuth <= 0.75 * np.pi)] = -np.tan(
            np.pi / 2 - azimuth[(azimuth > 0.25 * np.pi) & (azimuth <= 0.75 * np.pi)]
        )
        ddy[azimuth > 0.75 * np.pi] = 1

        # Shading routine
        z_flat = grid["z_2D"].ravel()
        z_current = z_flat[grid["ind"]]
        shade = np.zeros(grid["gpsum"], dtype=int)
        max_count = 200

        # Precompute grid indices and deltas ahead of time
        delta_y, delta_x = np.round(ddx).astype(int), np.round(ddy).astype(int)

        # Precompute flattened grid distances
        ddx_scaled = ddx * grid["dx"]
        ddy_scaled = ddy * grid["dx"]
        distance_grid = np.sqrt(ddx_scaled**2 + ddy_scaled**2)
        distance_scaled = np.outer(np.arange(1, max_count + 1), distance_grid)

        for count in range(1, max_count + 1):
            kk = np.clip(grid["yind"] + delta_y * count, 0, yl - 1)
            ll = np.clip(grid["xind"] + delta_x * count, 0, xl - 1)
            ind_kkll = ll * yl + kk

            elev_diff = z_flat[ind_kkll] - z_current
            gridangle = np.arctan(elev_diff / distance_scaled[count - 1])

            # Update shade array where elevation angle is exceeded
            shade[elevationangle <= gridangle] = 1

            # Break the loop if all cells are shaded
            if np.all(shade == 1):
                break

        # Build the final shade 2D array
        shade_2D = np.ones((xl, yl), dtype=int)
        shade_2D.flat[grid["ind"]] = shade
        OUT["shade"] = shade_2D.flatten()[grid["mask_2D"].flatten() == 1]
    else:
        # TODO: Shading based on lookup tables from Elmer/Ice
        # For now: assume flat surface and no shading by surrounding terrain
        OUT["shade"] = np.zeros_like(grid["x"])
        OUT["shade"][elevationangle < 0] = 1

    return OUT
