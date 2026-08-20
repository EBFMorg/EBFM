# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
from netCDF4 import Dataset, date2num

from .LOOP_general_functions import is_first_time_step, is_final_time_step

_FILL_VALUE = -9999.0


def _write_unstructured_grid(nc_file, grid):
    """Write unstructured mesh coordinates and topology to an open NetCDF file."""
    n_cells = grid["x"].shape[0]

    nc_file.createDimension("cell", n_cells)

    x = nc_file.createVariable("x", np.float64, ("cell",))
    y = nc_file.createVariable("y", np.float64, ("cell",))
    z = nc_file.createVariable("z", np.float64, ("cell",))
    lon = nc_file.createVariable("lon", np.float64, ("cell",))
    lat = nc_file.createVariable("lat", np.float64, ("cell",))

    x[:] = grid["x"]
    y[:] = grid["y"]
    z[:] = grid["z"]
    lon[:] = grid["lon"]
    lat[:] = grid["lat"]

    x.units = "m"
    y.units = "m"
    z.units = "m"
    lon.units = "degrees_east"
    lat.units = "degrees_north"

    x.description = "Cell-center x coordinate"
    y.description = "Cell-center y coordinate"
    z.description = "Cell-center surface elevation"
    lon.description = "Cell-center longitude"
    lat.description = "Cell-center latitude"

    mesh = grid.get("mesh")
    if mesh is None:
        return

    n_vertices = mesh.x_vertices.shape[0]
    vertices_per_cell = mesh.cell_to_vertex.shape[1]

    nc_file.createDimension("vertex", n_vertices)
    nc_file.createDimension("nv", vertices_per_cell)

    vx = nc_file.createVariable("x_vertex", np.float64, ("vertex",))
    vy = nc_file.createVariable("y_vertex", np.float64, ("vertex",))
    vz = nc_file.createVariable("z_vertex", np.float64, ("vertex",))
    vlon = nc_file.createVariable("lon_vertex", np.float64, ("vertex",))
    vlat = nc_file.createVariable("lat_vertex", np.float64, ("vertex",))
    cell_vertices = nc_file.createVariable("cell_vertices", np.int32, ("cell", "nv"))

    vx[:] = mesh.x_vertices
    vy[:] = mesh.y_vertices
    vz[:] = mesh.z_vertices
    vlon[:] = np.degrees(mesh.lon_vertices)
    vlat[:] = np.degrees(mesh.lat_vertices)
    cell_vertices[:, :] = mesh.cell_to_vertex.astype(np.int32)

    vx.units = "m"
    vy.units = "m"
    vz.units = "m"
    vlon.units = "degrees_east"
    vlat.units = "degrees_north"

    cell_vertices.description = "Zero-based local vertex indices for each unstructured cell"


def _write_structured_grid(nc_file, grid):
    """Write structured grid dimensions and coordinates to an open NetCDF file."""
    nc_file.createDimension("y", grid["x_2D"].shape[0])
    nc_file.createDimension("x", grid["x_2D"].shape[1])

    x = nc_file.createVariable("x", np.float64, ("y", "x"))
    y = nc_file.createVariable("y", np.float64, ("y", "x"))

    x[:, :] = grid["x_2D"]
    y[:, :] = grid["y_2D"]

    x.units = "m"
    y.units = "m"

    x.description = "Structured grid x coordinate"
    y.description = "Structured grid y coordinate"

    if "z_2D" in grid:
        z = nc_file.createVariable("z", np.float64, ("y", "x"))
        z[:, :] = grid["z_2D"]
        z.units = "m"
        z.description = "Structured grid surface elevation"

    if "lon_2D" in grid and "lat_2D" in grid:
        lon = nc_file.createVariable("lon", np.float64, ("y", "x"))
        lat = nc_file.createVariable("lat", np.float64, ("y", "x"))
        lon[:, :] = grid["lon_2D"]
        lat[:, :] = grid["lat_2D"]
        lon.units = "degrees_east"
        lat.units = "degrees_north"
        lon.description = "Structured grid longitude"
        lat.description = "Structured grid latitude"


def _output_dimensions(varname, grid):
    """Return NetCDF dimensions/chunking for a model output variable."""
    if grid["is_unstructured"]:
        if varname.startswith("sub"):
            return ("time", "cell", "nl"), (1, grid["x"].shape[0], grid["nl"])
        return ("time", "cell"), (1, grid["x"].shape[0])

    if varname.startswith("sub"):
        return ("time", "y", "x", "nl"), (1, grid["x_2D"].shape[0], grid["x_2D"].shape[1], grid["nl"])
    return ("time", "y", "x"), (1, grid["x_2D"].shape[0], grid["x_2D"].shape[1])


def _write_output_variable(nc_file, varname, var_1D, time_index, grid):
    """Write one model output variable to an open NetCDF file."""
    if grid["is_unstructured"]:
        if varname.startswith("sub"):
            nc_file[varname][time_index, :, :] = var_1D
        else:
            nc_file[varname][time_index, :] = var_1D
        return

    if varname.startswith("sub"):
        var_3D = np.full((grid["x_2D"].size, grid["nl"]), _FILL_VALUE)
        var_3D[grid["ind"], :] = var_1D
        var_4D = var_3D.reshape(-1, grid["nl"]).reshape(*grid["x_2D"].shape, grid["nl"])
        nc_file[varname][time_index, :, :, :] = var_4D
    else:
        var_2D = np.full(grid["x_2D"].shape, _FILL_VALUE)
        var_2D.flat[grid["ind"]] = var_1D
        nc_file[varname][time_index, :, :] = var_2D


def main(OUTFILE, io, OUT, grid, t, time):
    # Specify variables to be written
    if is_first_time_step(t):
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
        # "shade" is written as a fraction but sourced from the boolean OUT["is_shaded"]
        source_key = "is_shaded" if varname == "shade" else varname
        temp_long = np.float64(OUT[source_key])

        # Initialize TEMP storage
        if t % io["freqout"] == 0:
            OUTFILE.setdefault("TEMP", {})
            OUTFILE["TEMP"][varname] = np.zeros_like(temp_long)

        # Handle type: sample, mean, or sum
        if var_type == "sample":
            if (t + 1 + io["freqout"] // 2) % io["freqout"] == 0:  # TODO: correct?
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
        if is_first_time_step(t):
            if not os.path.exists(io["outdir"]):
                os.makedirs(io["outdir"])

            io["fid"] = {}
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                filepath = os.path.join(io["outdir"], f"OUT_{varname}.bin")
                io["fid"][varname] = open(filepath, "wb")

        # Write variables to binary files when `freqout` is met
        if (t + 1) % io["freqout"] == 0:  # TODO: correct?
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                OUTFILE[varname] = OUTFILE["TEMP"][varname]
                io["fid"][varname].write(OUTFILE[varname].astype("float32").tobytes())

        # Close all file streams and save run metadata at the final time step
        if is_final_time_step(t, time):
            for file in io["fid"].values():
                file.close()

            # TODO: WORK IN PROGRESS
            # Prepare the runinfo dictionary
            # runinfo = {"grid": grid, "time": time, "IOout": io, "Cout": C}

        return True

    def save_netCDF_file():
        """
        Save model output stored in OUTFILE to a NetCDF file.

        Structured grids are written on dimensions (time, y, x).
        Unstructured grids are written on dimensions (time, cell), with
        optional mesh topology if grid["mesh"] is available.
        """
        # Epoch for time variable
        time_units = "days since 1970-01-01 00:00:00"
        time_calendar = "gregorian"

        # Initialize NetCDF file at the first time step
        if is_first_time_step(t):
            if not os.path.exists(io["outdir"]):
                os.makedirs(io["outdir"])

            # Create NetCDF file
            nc_filepath = os.path.join(io["outdir"], "model_output.nc")
            io["nc_file"] = Dataset(nc_filepath, "w", format="NETCDF4")
            io["nc_file"].createDimension("time", None)  # Unlimited time dimension

            if grid["is_unstructured"]:
                _write_unstructured_grid(io["nc_file"], grid)
            else:
                _write_structured_grid(io["nc_file"], grid)

            io["nc_file"].createDimension("nl", grid["nl"])  # Vertical layers for `sub` variables

            # Define standard output variables
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                var_units = entry[1]
                var_desc = entry[3]

                dimensions, chunksizes = _output_dimensions(varname, grid)

                nc_var = io["nc_file"].createVariable(
                    varname=varname,
                    datatype=np.float32,
                    dimensions=dimensions,
                    zlib=True,
                    complevel=4,
                    fill_value=_FILL_VALUE,
                    chunksizes=chunksizes,
                )

                # Assign metadata
                nc_var.units = var_units
                nc_var.description = var_desc

            # Define a time variable to track simulation steps
            nc_time = io["nc_file"].createVariable("time", np.float64, ("time",), zlib=True, fill_value=_FILL_VALUE)
            nc_time.units = time_units
            nc_time.calendar = time_calendar
            nc_time.description = "Time at which data is recorded, in days since 1970-01-01 00:00:00"

        # Write data to NetCDF at the specified frequency
        if (t + 1) % io["freqout"] == 0:  # TODO: correct?
            time_index = t // io["freqout"]

            # Calculate time in "days since 1970-01-01"
            time_days_since_1970 = date2num(time["TCUR"], units=time_units, calendar=time_calendar)

            # Write time variable
            io["nc_file"]["time"][time_index] = time_days_since_1970

            # Write variables to NetCDF
            for entry in OUTFILE["varsout"]:
                varname = entry[0]
                var_1D = OUTFILE["TEMP"][varname]
                _write_output_variable(io["nc_file"], varname, var_1D, time_index, grid)

        # Close the NetCDF file at the final time step
        if is_final_time_step(t, time):
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
