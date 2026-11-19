# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read a structured grid from a NetCDF file."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset
from numpy import dtype, ndarray
from pyproj import Transformer

logger = logging.getLogger(__name__)


def read_NETCDF_grid(gridfile: Path, fill_missing_elevation: bool = False):
    """
    Read grid information from an NetCDF (.nc) file with DEM.

    Cells with missing orography are always inactive. Their elevation is NaN, or 0 m if `fill_missing_elevation`.

    Parameters:
        gridfile: Path to NetCDF file (.nc).
        fill_missing_elevation: Set missing elevations to 0 m instead of NaN.

    Returns:
        dict: A dictionary named `input_data` containing:
            - 'x': 2D NumPy array of UTM zone 24N easting coordinates (m).
            - 'y': 2D NumPy array of UTM zone 24N northing coordinates (m).
            - 'z': 2D NumPy array of orography/elevation.
            - 'mask': 2D NumPy array of fractional glacier mask values.
            - 'lon': 2D NumPy array of longitude coordinates in degrees East.
            - 'lat': 2D NumPy array of latitude coordinates.
    """

    logger.info("EBFM: Reading grid data from NetCDF file...")
    input_data: dict[str, ndarray[tuple[int, ...], dtype[Any]]] = {}

    try:
        with Dataset(gridfile, "r") as ncfile:
            if "orography" in ncfile.variables:
                raw_orography = ncfile.variables["orography"][:]
            elif "orog" in ncfile.variables:
                raw_orography = ncfile.variables["orog"][:]
            else:
                raise KeyError(f"Neither `orography` nor `orog` found in NetCDF grid file {gridfile}.")

            if "mask" not in ncfile.variables:
                raise KeyError(f"Missing `mask` in in NetCDF grid file {gridfile}.")

            raw_mask = ncfile.variables["mask"][:]
            latitude = np.asarray(ncfile.variables["latitude"][:])
            longitude = np.asarray(ncfile.variables["longitude"][:])

            orography = np.asarray(raw_orography)
            mask = np.asarray(raw_mask, dtype=np.float32)

            orography = np.squeeze(orography)
            mask = np.squeeze(mask)
            latitude = np.squeeze(latitude)
            longitude = np.squeeze(longitude)

            if np.ma.isMaskedArray(raw_orography):
                valid_orography = ~np.squeeze(np.ma.getmaskarray(raw_orography))
            else:
                valid_orography = np.isfinite(orography)

            if latitude.ndim == 1 and longitude.ndim == 1:
                longitude, latitude = np.meshgrid(longitude, latitude)

            if orography.shape != mask.shape or orography.shape != latitude.shape or orography.shape != longitude.shape:
                raise ValueError(
                    "`orography`/`orog`, `mask`, `latitude`, and `longitude` must have matching 2D shapes. "
                    f"Got orography={orography.shape}, mask={mask.shape}, "
                    f"latitude={latitude.shape}, longitude={longitude.shape}."
                )

            mask = np.where(valid_orography & (mask > 0.0), 1.0, 0.0).astype(np.float32)
            if fill_missing_elevation:
                n_missing = int((~valid_orography).sum())
                if n_missing > 0:
                    logger.warning(f"Setting {n_missing} missing elevations in {gridfile} to 0 m.")
                orography = np.where(valid_orography, orography, 0.0)
            else:
                orography = np.where(valid_orography, orography, np.nan)
            longitude = np.mod(longitude, 360.0)

            if not np.any(mask > 0.0):
                raise ValueError(f"Mask of {gridfile} contains no active cells with `mask > 0`.")

            lonlat_to_utm24n = Transformer.from_crs("EPSG:4326", "EPSG:32624", always_xy=True)
            x, y = lonlat_to_utm24n.transform(longitude, latitude)

            input_data["x"] = x
            input_data["y"] = y
            input_data["z"] = orography
            input_data["lat"] = latitude
            input_data["lon"] = longitude
            input_data["mask"] = mask

    except FileNotFoundError:
        logger.info(f"File not found: {gridfile}")
        raise
    except KeyError as e:
        logger.info(f"Missing field in NetCDF file {gridfile}: {e}")
        raise

    return input_data
