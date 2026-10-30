# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from netCDF4 import Dataset, date2num
from pathlib import Path
import numpy as np

import logging

from .comm.gather import ColumnGather
from .restart import COLUMN_ID_VARIABLE, RESTART_VARIABLES

logger = logging.getLogger(__name__)


def main(OUT, io, restartdir: Path, gather: ColumnGather):
    """
    Write the restart file (boot file) if io["writebootfile"] is set.

    Collective over gather.comm: the root rank gathers the columns of all ranks one variable at a time and writes
    them, ordered by column id, together with their ids. Sets OUT["timelastsnow_netCDF"] on every rank, also when no
    restart file is written.

    @param[in] OUT model state of this rank's columns
    @param[in] io I/O settings: writebootfile and bootfileout
    @param[in] restartdir directory of the restart files, created if it does not exist
    @param[in] gather the columns of all ranks
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

    ncfile = None
    if gather.is_root:
        if restartdir:
            os.makedirs(restartdir, exist_ok=True)
        ncfile = Dataset(io["bootfileout"], "w", format="NETCDF4")

    try:
        for var_name in RESTART_VARIABLES:
            values = gather.gather(OUT[var_name])
            if ncfile is not None:
                _write_variable(ncfile, var_name, values)
        if ncfile is not None:
            _write_variable(ncfile, COLUMN_ID_VARIABLE, gather.column_ids)
    finally:
        if ncfile is not None:
            ncfile.close()

    if gather.is_root:
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
