# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from netCDF4 import Dataset, date2num
from pathlib import Path
import numpy as np

import logging

logger = logging.getLogger(__name__)


def main(OUT, io, restartdir: Path):
    def create_boot_file():
        """
        Create a restart file (boot file) and save it as a NetCDF file.

        Parameters:
        - OUT: Dictionary containing the data to be saved.
        @param[in] restartdir Path to the directory containing restart files (if initializing from restart file).
        Returns:
        - None
        """

        # Create a reboot directory if it does not exist
        if restartdir:
            os.makedirs(restartdir, exist_ok=True)

        # Check if we should write the boot file
        if io["writebootfile"]:
            # Define the output NetCDF file path
            # Create a new NetCDF file to store the boot variables
            with Dataset(io["bootfileout"], "w", format="NETCDF4") as ncfile:
                # Save each variable in the OUT dictionary to the NetCDF file
                for var_name, var_data in OUT.items():
                    # Handle different variable dimensions
                    if isinstance(var_data, np.ndarray):  # If it's a NumPy array
                        # Create the appropriate dimension(s) if not already defined
                        dims = []
                        for dimsize in var_data.shape:
                            dim_name = f"{var_name}_dim{len(dims)}"
                            if dim_name not in ncfile.dimensions:
                                ncfile.createDimension(dim_name, dimsize)
                            dims.append(dim_name)

                        # Create the variable
                        nc_var = ncfile.createVariable(var_name, var_data.dtype, tuple(dims))
                        nc_var[:] = var_data  # Write the data
                    elif np.isscalar(var_data):  # If it's a scalar, store it as a 0D variable
                        nc_var = ncfile.createVariable(var_name, type(var_data), ())
                        nc_var.assignValue(var_data)
                    else:
                        raise ValueError(f"Unsupported data type for variable: {var_name}")

            logger.info(f"Boot file saved to {io['bootfileout']}")

    OUT["timelastsnow_netCDF"] = date2num(
        OUT["timelastsnow"],
        units="days since 1970-01-01 00:00:00",
        calendar="gregorian",
    )
    OUT = {
        "subZ": OUT["subZ"],
        "subW": OUT["subW"],
        "subD": OUT["subD"],
        "subS": OUT["subS"],
        "subT": OUT["subT"],
        "subTmean": OUT["subTmean"],
        "snowmass": OUT["snowmass"],
        "Tsurf": OUT["Tsurf"],
        "ys": OUT["ys"],
        "timelastsnow_netCDF": OUT["timelastsnow_netCDF"],
        "alb_snow": OUT["alb_snow"],
        "h": OUT["h"],
        "x": OUT["x"],
        "y": OUT["y"],
    }

    # Create the boot file
    create_boot_file()
