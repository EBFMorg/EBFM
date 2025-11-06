# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
from netCDF4 import Dataset, date2num


def main(OUTFILE, io, OUT, grid, t, time, C):
    # Specify variables to be written
    if t == 1:
        OUTFILE["varsout"] = [
            ["smb", "m w.e.", "sum", "Climatic mass balance"],
            ["Tsurf", "K", "mean", "Surface temperature"],
            ["climT", "K", "mean", "Air temperature"],
            ["climP", "m w.e.", "sum", "Precipitation"],
            ["climC", "fraction", "mean", "Cloud cover"],
            ["climRH", "fraction", "mean", "Relative humidity"],
            ["climWS", "m s-1", "mean", "Wind speed"],
            ["climPres", "Pa", "mean", "Air pressure"],
            ["climrain", "m w.e.", "sum", "Rainfall"],
            ["climsnow", "m w.e.", "sum", "Snowfall"],
            ["snowmass", "m w.e.", "mean", "Snow mass"],
            ["smb_cumulative", "m w.e.", "mean", "Cumulative mass balance"],
            ["melt", "m w.e.", "sum", "Melt"],
            ["refr", "m w.e.", "sum", "Refreezing"],
            ["runoff", "m w.e.", "sum", "Runoff"],
            ["runoff_surf", "m w.e.", "sum", "Surface runoff"],
            ["runoff_slush", "m w.e.", "sum", "Slush runoff"],
            ["SWin", "W m^-2", "mean", "Incoming SW radiation"],
            ["SWout", "W m^-2", "mean", "Reflected SW radiation"],
            ["LWin", "W m^-2", "mean", "Incoming LW radiation"],
            ["LWout", "W m^-2", "mean", "Outgoing LW radiation"],
            ["SHF", "W m^-2", "mean", "Sensible heat flux"],
            ["LHF", "W m^-2", "mean", "Latent heat flux"],
            ["GHF", "W m^-2", "mean", "Subsurface heat flux"],
            ["surfH", "m", "sample", "Surface height"],
            ["albedo", "fraction", "mean", "Albedo"],
            ["shade", "fraction", "mean", "Shading (0=not shaded, 1=shaded)"],
            ["subD", "kg m^-3", "sample", "Density"],
            ["subT", "K", "sample", "Temperature"],
            ["subS", "mm w.e.", "sample", "Slush water content"],
            ["subW", "mm w.e.", "sample", "Irreducible water"],
            ["subZ", "m", "sample", "Layer thickness"],
        ]

        io["varsout"] = [
            {"varname": v[0], "units": v[1], "type": v[2], "description": v[3]} for v in OUTFILE["varsout"]
        ]

    # Update OUTFILE.TEMP with variables to be stored
    for entry in OUTFILE["varsout"]:
        varname, var_type = entry[0], entry[2]
        temp_long = np.float64(OUT[varname])

        # Initialize TEMP storage
        if (t - 1) % io["freqout"] == 0:
            OUTFILE.setdefault("TEMP", {})
            OUTFILE["TEMP"][varname] = np.zeros_like(temp_long)

        # Handle type: sample, mean, or sum
        if var_type == "sample":
            if (t + io["freqout"] // 2) % io["freqout"] == 0:
                OUTFILE["TEMP"][varname] = temp_long
        elif var_type == "mean":
            OUTFILE["TEMP"][varname] += temp_long / io["freqout"]
        elif var_type == "sum":
            OUTFILE["TEMP"][varname] += temp_long

    def save_binary_files():
        """
        Write model output to binary files and save run information.

        Parameters:
        - OUTFILE: Dictionary storing output details and temporary data.
        - io: Dictionary holding I/O parameters, e.g., freqout, outdir.
        - OUT: Dictionary with variables to be output.
        - grid: Grid information.
        - t: Current time step.
        - time: Dictionary containing time-related variables.
        - C: Dictionary of constants.

        Returns:
        - Updated OUTFILE and io dictionaries.
        """

        # Save output to binary files at the first time step
        if t == 1:
            if not os.path.exists(io["outdir"]):
                os.makedirs(io["outdir"])

            io["fid"] = {}
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                filepath = os.path.join(io["outdir"], f"OUT_{varname}.bin")
                io["fid"][varname] = open(filepath, "wb")

        # Write variables to binary files when `freqout` is met
        if t % io["freqout"] == 0:
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                OUTFILE[varname] = OUTFILE["TEMP"][varname]
                io["fid"][varname].write(OUTFILE[varname].astype("float32").tobytes())

        # Close all file streams and save run metadata at the final time step
        if t == time["tn"]:
            for file in io["fid"].values():
                file.close()

            # TODO: WORK IN PROGRESS
            # Prepare the runinfo dictionary
            # runinfo = {"grid": grid, "time": time, "IOout": io, "Cout": C}

        return True

    def save_netCDF_file():
        """
        Save model output stored in OUTFILE to a NetCDF file. Converts 1D data to 2D grids.

        Parameters:
        - OUTFILE: Dictionary storing output details and temporary data.
        - io: Dictionary holding I/O parameters, e.g., freqout, outdir.
        - grid: Grid information (e.g., x_2D size and ind: mapping 1D -> 2D).
        - t: Current time step.
        - time: Dictionary containing time-related variables.
        - freqout: Frequency of output (save interval).

        Returns:
        - Updated io dictionary with NetCDF file reference.
        """
        # Epoch for time variable
        time_units = "days since 1970-01-01 00:00:00"
        time_calendar = "gregorian"

        # Initialize NetCDF file at the first time step
        if t == 1:
            if not os.path.exists(io["outdir"]):
                os.makedirs(io["outdir"])

            # Create NetCDF file
            nc_filepath = os.path.join(io["outdir"], "model_output.nc")
            io["nc_file"] = Dataset(nc_filepath, "w", format="NETCDF4")

            # Define dimensions
            io["nc_file"].createDimension("time", None)  # Unlimited time dimension
            io["nc_file"].createDimension("y", grid["x_2D"].shape[0])  # 2D grid rows
            io["nc_file"].createDimension("x", grid["x_2D"].shape[1])  # 2D grid columns
            io["nc_file"].createDimension("nl", grid["nl"])  # Vertical layers for `sub` variables

            # Define standard output variables
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                var_units = entry[1]
                var_desc = entry[3]

                # Check if variable is a `sub` variable
                if varname.startswith("sub"):
                    # Define variable as 4D: (time, y, x, nl)
                    nc_var = io["nc_file"].createVariable(
                        varname,
                        np.float32,
                        ("time", "y", "x", "nl"),
                        zlib=True,
                        complevel=4,
                        fill_value=-9999.0,  # Fill missing values
                        chunksizes=(
                            1,
                            grid["x_2D"].shape[0],
                            grid["x_2D"].shape[1],
                            grid["nl"],
                        ),
                    )
                else:
                    # Define variable as 3D: (time, y, x)
                    nc_var = io["nc_file"].createVariable(
                        varname,
                        np.float32,
                        ("time", "y", "x"),
                        zlib=True,
                        complevel=4,
                        fill_value=-9999.0,
                        chunksizes=(1, grid["x_2D"].shape[0], grid["x_2D"].shape[1]),
                    )

                # Assign metadata
                nc_var.units = var_units
                nc_var.description = var_desc

            # Define a time variable to track simulation steps
            nc_time = io["nc_file"].createVariable("time", np.float64, ("time",), zlib=True, fill_value=-9999.0)
            nc_time.units = time_units
            nc_time.calendar = time_calendar
            nc_time.description = "Time at which data is recorded, in days since 1970-01-01 00:00:00"

        # Write data to NetCDF at the specified frequency (e.g., for average/summed values)
        if t % io["freqout"] == 0:
            time_index = t // io["freqout"] - 1  # Determine the time slice index

            # Calculate time in "days since 1970-01-01"
            time_days_since_1970 = date2num(time["TCUR"], units=time_units, calendar=time_calendar)

            # Write time variable
            io["nc_file"]["time"][time_index] = time_days_since_1970

            # Write variables to NetCDF
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                var_1D = OUTFILE["TEMP"][varname]  # 1D data

                # Handle `sub` variables (4D: time, y, x, nl)
                if varname.startswith("sub"):
                    var_3D = np.full((grid["x_2D"].size, grid["nl"]), -9999.0)
                    var_3D[grid["ind"], :] = var_1D
                    var_4D = var_3D.reshape(-1, grid["nl"]).reshape(*grid["x_2D"].shape, grid["nl"])
                    io["nc_file"][varname][time_index, :, :, :] = var_4D
                else:
                    var_2D = np.full(grid["x_2D"].shape, -9999.0)
                    var_2D.flat[grid["ind"]] = var_1D
                    io["nc_file"][varname][time_index, :, :] = var_2D

        # Close the NetCDF file at the final time step
        if t == time["tn"]:
            io["nc_file"].close()

        return True

    # Save output as binary or netCDF files
    if io["output_type"] == 1:
        save_binary_files()
    elif io["output_type"] == 2:
        save_netCDF_file()
    else:
        print("Invalid output type. Please choose 1 for binary files or 2 for NetCDF.")

    return io, OUTFILE
