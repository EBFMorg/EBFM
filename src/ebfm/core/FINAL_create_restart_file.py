# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from netCDF4 import Dataset, date2num
from pathlib import Path
import numpy as np

import logging

from .restart import COLUMN_ID_VARIABLE, RESTART_VARIABLES

logger = logging.getLogger(__name__)


def main(OUT, io, column_ids: np.ndarray):
    """
    Write the restart file (boot file) of this rank's columns if io["writebootfile"] is set.

    Writes the restart variables together with the ids of the columns. Sets OUT["timelastsnow_netCDF"], also when no
    restart file is written.

    @param[in] OUT model state of this rank's columns
    @param[in] io I/O settings: writebootfile and bootfileout, whose directory is created if it does not exist
    @param[in] column_ids ids of this rank's columns (grid["column_id"])
    """
    # float64 even if all times are whole days, for which date2num returns integers
    OUT["timelastsnow_netCDF"] = np.asarray(
        date2num(
            OUT["timelastsnow"],
            units="days since 1970-01-01 00:00:00",
            calendar="gregorian",
        ),
        dtype=np.float64,
    )

    if not io["writebootfile"]:
        return

    os.makedirs(Path(io["bootfileout"]).parent, exist_ok=True)
    with Dataset(io["bootfileout"], "w", format="NETCDF4") as ncfile:
        for var_name in RESTART_VARIABLES:
            _write_variable(ncfile, var_name, OUT[var_name])
        _write_variable(ncfile, COLUMN_ID_VARIABLE, np.asarray(column_ids))

    logger.info(f"Boot file saved to {io['bootfileout']}")


def _write_variable(ncfile: Dataset, var_name: str, var_data: np.ndarray) -> None:
    """Write an array to the open restart file, with one dimension of its own per axis."""
    if not isinstance(var_data, np.ndarray):
        raise ValueError(f"Unsupported data type for variable: {var_name}")

    dims = []
    for dimsize in var_data.shape:
        dim_name = f"{var_name}_dim{len(dims)}"
        if dim_name not in ncfile.dimensions:
            ncfile.createDimension(dim_name, dimsize)
        dims.append(dim_name)

    nc_var = ncfile.createVariable(var_name, var_data.dtype, tuple(dims))
    nc_var[:] = var_data
