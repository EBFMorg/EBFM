# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from netCDF4 import Dataset, date2num
import numpy as np


def main(OUT, io):

    def create_boot_file():
        """
        Create a restart file (boot file) and save it as a NetCDF file.

        Parameters:
        - OUT: Dictionary containing the data to be saved.
        - io: Dictionary containing I/O parameters including reboot directory and output filename.

        Returns:
        - None
        """
        # Create a reboot directory if it does not exist
        if not os.path.exists(io["rebootdir"]):
            os.makedirs(io["rebootdir"])

        # Check if we should write the boot file
        if io.get("writebootfile", False):
            # Define the output NetCDF file path
            boot_file_path = os.path.join(io["rebootdir"], io["bootfileout"])

            # Create a new NetCDF file to store the boot variables
            with Dataset(boot_file_path, "w", format="NETCDF4") as ncfile:
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

            print(f"Boot file saved to {boot_file_path}")

            return True

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
    }

    # Create the boot file
    create_boot_file()
