# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any
import numpy as np
from numpy import ndarray, dtype
import scipy.io as sio
from pyproj import Transformer
from netCDF4 import Dataset, num2date

from pathlib import Path
from reader import read_elmer_mesh, read_dem, read_dem_xios

from elmer.mesh import Mesh
from ebfm.config import GridConfig
from ebfm.grid import GridInputType

from ebfm.constants.materials import Ice, FreshSnow


import logging

logger = logging.getLogger(__name__)


def init_config():
    """
    Set model parameters, I/O, and physics settings.

    @param[in] time_config Time configuration object.

    @returns:
        io (dict): Input/output parameters.
        phys (dict): Model physics settings.
    """

    # ---------------------------------------------------------------------
    # Model physics
    # ---------------------------------------------------------------------
    phys = {}
    phys["percolation"] = "normal"  # Water percolation scheme
    # Options:
    #   - 'bucket': tipping-bucket method (all water added at the surface)
    #   - 'normal': normally distributed deep percolation
    #   - 'linear': linearly distributed deep percolation
    #   - 'uniform': uniformly distributed deep percolation

    phys["snow_compaction"] = "firn+snow"  # Snow and firn compaction scheme
    # Options:
    #   - 'firn_only': apply Ligtenberg et al. (2011) for all snow and firn layers
    #   - 'firn+snow': apply Ligtenberg et al. (2011) for firn and Kampenhout et al. (2017) for seasonal snow

    # ---------------------------------------------------------------------
    # Input/output parameters
    # ---------------------------------------------------------------------
    io = {}

    io["homedir"] = os.getcwd()  # Home directory
    io["outdir"] = os.path.join(io["homedir"], "Output")  # Output directory
    io["rebootdir"] = os.path.join(io["homedir"], "Reboot")  # Restart file directory
    io["readbootfile"] = False  # REBOOT: read initial conditions from file (True/False)
    io["writebootfile"] = True  # REBOOT: write file for rebooting (True/False)
    io["bootfilein"] = "boot_final.nc"  # REBOOT: bootfile to be read
    io["bootfileout"] = "boot_final.nc"  # REBOOT: bootfile to be written
    io["freqout"] = 8  # OUTPUT: frequency of storing output (every n-th time-step)
    io["output_type"] = 2  # Set output file type: 1 = binary files, 2 = netCDF file

    # Ensure output and reboot directories exist
    os.makedirs(io["outdir"], exist_ok=True)
    os.makedirs(io["rebootdir"], exist_ok=True)

    # Return the initialized parameters
    return io, phys


def init_constants():
    """
    Initializes the model constants.
    Returns:
        C (dict): Dictionary containing constant values used in the simulation.
    """

    C = {}

    # ---------------------------------------------------------------------
    # Energy balance model
    # ---------------------------------------------------------------------
    C["tstar_wet"] = 15  # Albedo decay time-scale wet snow (days)
    C["tstar_dry"] = 30  # Albedo decay time-scale dry snow (days)
    C["tstar_K"] = 7  # Albedo decay time-scale coefficient
    C["dstar"] = 7.0  # Albedo characteristic depth (mm w.e.)
    C["b"] = 0.455  # Constant in LWin formulation
    C["ecl"] = 0.960  # Constant in LWin formulation
    C["p"] = 2  # Exponent in LWin formulation
    C["boltz"] = 5.67e-8  # Stefan-Boltzmann constant (W m-2 K-4)
    C["VP0"] = 610.5  # Vapor pressure at 0 degrees C
    C["Cp"] = 1005.7  # Specific heat of dry air (J kg-1 K-1)
    C["Cw"] = 4187.0  # Specific heat of water (J kg-1 K-1)
    C["Ls"] = 2.83e6  # Latent heat of sublimation/riming (J kg-1)
    C["Lm"] = 0.33e6  # Latent heat of melting/fusion (J kg-1)
    C["Lv"] = 2.5e6  # Latent heat of evaporation/condensation (J kg-1)
    C["Rv"] = 462.0  # Specific gas constant water vapor (J kg-1 K-1)
    C["Rd"] = 287.0  # Specific gas constant dry air (J kg-1 K-1)
    C["eps"] = 0.622  # C.Rd/C.Rv
    C["dTacc"] = 0.01  # Threshold dT in solving the energy balance equation (K)
    C["Pref"] = 1015e2  # Reference air pressure (Pa)
    C["Pr"] = 5  # Prandtl number in SHF/LHF formulation
    C["T0"] = 273.15  # Melting temperature of ice (K)
    C["g"] = 9.81  # Gravitational acceleration (m s-2)
    C["rd"] = 8.314  # Universal gas constant (J mol-1 K-1)
    C["k_aer"] = 0.974  # Aerosol transmissivity exponent
    C["k_turb"] = 0.0004  # Turbulent flux coefficient
    C["turb"] = 0.0025  # Background turbulent exchange coefficient
    C["rainsnowT"] = 273.75  # Temperature of snow to rain transition (K)
    C["Pthres"] = 2.5e-8  # Threshold precipitation to reset time since last snowfall (m w.e. s-1)

    # ---------------------------------------------------------------------
    # Snow model
    # ---------------------------------------------------------------------
    C["Ec"] = 60000  # Gravitational densification factor
    C["Eg"] = 42400  # Gravitational densification factor
    C["Trunoff"] = 0.001  # Slush runoff time-scale (days)
    C["geothermal_flux"] = 0.0  # Geothermal heat flux (W m-2)
    C["perc_depth"] = 6.0  # Characteristic deep percolation depth (m)

    return C


def init_grid(config: GridConfig):
    # ---------------------------------------------------------------------
    # Grid parameters
    # ---------------------------------------------------------------------
    grid = {}
    grid["utmzone"] = 33  # UTM zone
    grid["max_subZ"] = 0.1  # Maximum first layer thickness (m)
    grid["nl"] = 50  # Number of vertical layers
    grid["doubledepth"] = True  # Double vertical layer depth at specified layers (True/False)
    grid["split"] = np.array([15, 25, 35])  # Vertical layer numbers at which layer depth doubles
    grid["is_partitioned"] = config.is_partitioned
    grid["is_unstructured"] = config.is_unstructured

    # Read grid from Elmer, elevations from BedMachine
    if config.dem_file:
        grid_input_type_supporting_dem = [GridInputType.CUSTOM, GridInputType.ELMERXIOS]
        assert (
            config.grid_type in grid_input_type_supporting_dem
        ), f"DEM file can only be specified for {grid_input_type_supporting_dem}."
        if config.is_partitioned:
            mesh: Mesh = read_elmer_mesh(
                mesh_root=config.mesh_file,
                is_partitioned=config.is_partitioned,
                partition_id=config.partition_id,
            )
        else:
            mesh: Mesh = read_elmer_mesh(mesh_root=config.mesh_file)

        grid["x"], grid["y"] = mesh.x_vertices, mesh.y_vertices
        if config.grid_type is GridInputType.CUSTOM:
            grid["z"] = read_dem(config.dem_file, grid["x"], grid["y"])
            grid["lat"] = np.zeros_like(grid["x"]) + 75  # test values!
            grid["lon"] = np.zeros_like(grid["x"]) + 320  # test values!
        if config.grid_type is GridInputType.ELMERXIOS:
            grid = read_dem_xios(config.dem_file, grid)
        grid["mask"] = np.ones_like(grid["x"])  # treats every grid cell as glacier
        grid["gpsum"] = np.sum(grid["mask"] == 1)  # number of modelled grid cells
        grid["slope_x"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_y"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_beta"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_gamma"] = np.zeros_like(grid["x"])  # test values!
        grid["mesh"] = mesh
        grid["has_shading"] = False  # TODO: see https://github.com/EBFMorg/EBFM/issues/11
        # TODO later add slope
        # dzdx, dzdy = mesh.dzdy, mesh.dzdy
    elif config.grid_type is GridInputType.ELMER:  # Read grid and elevations from Elmer
        mesh: Mesh = read_elmer_mesh(config.mesh_file)

        # assuming mesh/MESH/mesh.nodes contains DEM data in the z component
        # see mesh/README.md for the required preprocessing steps.
        grid["x"], grid["y"], grid["z"] = (
            mesh.x_vertices,
            mesh.y_vertices,
            mesh.z_vertices,
        )
        grid["z"] = np.random.uniform(0, 100, size=len(grid["x"]))  # test values!
        grid["slope_x"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_y"] = np.zeros_like(grid["x"])  # test values!
        grid["lat"] = np.zeros_like(grid["x"]) + 75  # test values!
        grid["lon"] = np.zeros_like(grid["x"]) + 320  # test values!
        grid["slope_beta"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_gamma"] = np.zeros_like(grid["x"])  # test values!
        grid["mask"] = np.ones_like(grid["x"])  # treats every grid cell as glacier
        grid["gpsum"] = np.sum(grid["mask"] == 1)  # number of modelled grid cells

        # TODO later add slope
        # grid["slope_x"], grid["slope_y"] = mesh.dzdy, mesh.dzdy
        grid["mesh"] = mesh
        grid["has_shading"] = False  # TODO: see https://github.com/EBFMorg/EBFM/issues/11
    elif config.grid_type is GridInputType.MATLAB:  # Read grid and elevations from example MATLAB file
        # ---------------------------------------------------------------------
        # Read and process grid information
        # ---------------------------------------------------------------------
        # Read grid data
        input_data = read_MATLAB_grid(config.mesh_file)
        grid["x_2D"] = input_data["x"][0][0]
        grid["y_2D"] = input_data["y"][0][0]
        grid["z_2D"] = input_data["z"][0][0]
        grid["mask_2D"] = input_data["mask"][0][0]

        # Determine domain extent
        grid["has_shading"] = True
        grid["Lx"], grid["Ly"] = grid["x_2D"].shape

        # Flip grid E-W or N-S when needed
        fy, _ = np.gradient(grid["y_2D"])

        if fy[0, 0] < 0:
            grid["x_2D"] = np.flipud(grid["x_2D"])
            grid["y_2D"] = np.flipud(grid["y_2D"])
            grid["z_2D"] = np.flipud(grid["z_2D"])
            grid["mask_2D"] = np.flipud(grid["mask_2D"])

        _, fx = np.gradient(grid["x_2D"])
        if fx[0, 0] < 0:
            grid["x_2D"] = np.fliplr(grid["x_2D"])
            grid["y_2D"] = np.fliplr(grid["y_2D"])
            grid["z_2D"] = np.fliplr(grid["z_2D"])
            grid["mask_2D"] = np.fliplr(grid["mask_2D"])

        # Calculate grid spacing
        grid["dx"] = grid["x_2D"][0][1] - grid["x_2D"][0][0]

        # Calculate number of modeled grid cells (gpsum)
        grid["gpsum"] = np.sum(grid["mask_2D"] == 1)

        # Create 1-D mask
        grid["mask"] = grid["mask_2D"][grid["mask_2D"] == 1]

        # Calculate latitude & longitude fields (from the original UTM coordinates)
        utmzone = grid["utmzone"]  # Assume this is already part of the grid
        utm_to_latlon = Transformer.from_crs(f"EPSG:{32600 + utmzone}", "EPSG:4326", always_xy=True)
        x_coords = grid["x_2D"].ravel()
        y_coords = grid["y_2D"].ravel()
        lon, lat = utm_to_latlon.transform(x_coords, y_coords)

        # Reshape to 2D arrays matching the input's shape
        grid["lon_2D"] = lon.reshape(grid["x_2D"].shape)
        grid["lat_2D"] = lat.reshape(grid["y_2D"].shape)

        # Store 1-D (vectorized) grid information
        mask_flat = grid["mask_2D"].flatten()
        grid["x"] = grid["x_2D"].flatten()[mask_flat == 1]
        grid["y"] = grid["y_2D"].flatten()[mask_flat == 1]
        grid["z"] = grid["z_2D"].flatten()[mask_flat == 1]
        grid["ind"] = np.where(grid["mask_2D"].flatten() == 1)
        grid["xind"], grid["yind"] = np.where(grid["mask_2D"] == 1)

        # ---------------------------------------------------------------------
        # Grid slope and aspect
        # ---------------------------------------------------------------------
        # Calculate slope and aspect
        gradN, gradE = np.gradient(grid["z_2D"], grid["y_2D"][:, 0], grid["x_2D"][0, :])
        slope_rad = np.arctan(np.sqrt(gradN**2 + gradE**2))
        slope_deg = np.degrees(slope_rad)
        aspect = np.arctan2(-gradE, -gradN) * (180 / np.pi)
        aspect[aspect < 0] = 360 + aspect[aspect < 0]
        grid["slope"] = np.tan(np.radians(slope_deg))
        grid["slope_x"] = gradE
        grid["slope_y"] = gradN
        grid["aspect"] = aspect
        grid["slope_2D"] = grid["slope"]
        grid["slope_beta_2D"] = np.arctan(grid["slope"])

        # Convert 2-D fields to 1-D vectors
        grid["slope"] = grid["slope"].flatten()[mask_flat == 1]
        grid["slope_x"] = grid["slope_x"].flatten()[mask_flat == 1]
        grid["slope_y"] = grid["slope_y"].flatten()[mask_flat == 1]
        grid["aspect"] = grid["aspect"].flatten()[mask_flat == 1]
        grid["lat"] = grid["lat_2D"].flatten()[mask_flat == 1]
        grid["lon"] = grid["lon_2D"].flatten()[mask_flat == 1]

        # Calculate slope_beta and slope_gamma (defining the tilt and orientation of a sloping surface)
        grid["slope_beta"] = np.arctan(grid["slope"])
        grid["slope_gamma"] = np.zeros_like(grid["slope"])
        grid["slope_gamma"][grid["slope_y"] > 0] = np.arctan(
            -grid["slope_x"][grid["slope_y"] > 0] / grid["slope_y"][grid["slope_y"] > 0]
        )
        cond = np.logical_and(grid["slope_y"] < 0, grid["slope_x"] > 0)
        grid["slope_gamma"][cond] = -np.pi + np.arctan(-grid["slope_x"][cond] / grid["slope_y"][cond])
        cond = np.logical_and(grid["slope_y"] < 0, grid["slope_x"] < 0)
        grid["slope_gamma"][cond] = np.pi + np.arctan(-grid["slope_x"][cond] / grid["slope_y"][cond])
        grid["slope_gamma"][(grid["slope_x"] == 0) & (grid["slope_y"] < 0)] = np.pi
        grid["slope_gamma"][(grid["slope_x"] == 0) & (grid["slope_y"] == 0)] = 0
        grid["slope_gamma"][(grid["slope_x"] > 0) & (grid["slope_y"] == 0)] = np.pi / 2
        grid["slope_gamma"][(grid["slope_x"] < 0) & (grid["slope_y"] == 0)] = -np.pi / 2
        grid["slope_gamma"] = -grid["slope_gamma"]
    elif grid["input_type"] is GridInputType.ELMER_XIOS:
        grid = read_elmer_xios_grid(grid=grid, gridfile=config.dem_file)
        grid["gpsum"] = grid["z"].shape[0]
        grid["mask"] = (grid["h"] > 1.0) * 1.0
        grid["x"] = np.zeros_like(grid["z"])
        grid["slope_beta"] = np.zeros_like(grid["x"])  # test values!
        grid["slope_gamma"] = np.zeros_like(grid["x"])  # test values!
    else:
        raise ValueError(f"Unsupported grid input type {config.grid_type} specified in configuration.")

    return grid


def read_elmer_xios_grid(grid, gridfile: Path):
    import netCDF4 as nc

    print(gridfile)
    with nc.Dataset(gridfile) as file:
        grid["lat"] = file["mesh2D_node_x"][:]
        grid["lon"] = file["mesh2D_node_y"][:]
        grid["x"] = file["x"][:]
        grid["y"] = file["y"][:]
        grid["z"] = np.squeeze(file["zs"][:].data)
        grid["h"] = np.squeeze(file["h"][:].data)
    return grid


def read_MATLAB_grid(gridfile: Path):
    """
    Provides grid information by reading from a .mat file or allowing user input.

    Parameters:
        gridfile: Path to the .mat file containing grid data.

    Returns:
        dict: A dictionary named `input_data` containing:
            - 'x': 2D NumPy array of UTM easting coordinates (m).
            - 'y': 2D NumPy array of UTM northing coordinates (m).
            - 'z': 2D NumPy array of elevation (m).
            - 'mask': 2D NumPy array of glacier mask (0 = no glacier, 1 = glacier).
    """

    logger.info("EBFM: Reading grid data from MATLAB file...")
    input_data: dict[str, ndarray[tuple[int, ...], dtype[Any]]] = {}

    try:
        # Load MATLAB file
        mat_file = sio.loadmat(str(gridfile))
        grid_svalbard = mat_file.get("grid_svalbard", None)

        if grid_svalbard is None:
            raise ValueError("`grid_svalbard` not found in the .mat file!")

        # Extract data from the MATLAB structure
        input_data["x"] = grid_svalbard["x"]
        input_data["y"] = grid_svalbard["y"]
        input_data["z"] = grid_svalbard["z"]
        input_data["mask"] = grid_svalbard["mask"]

    except FileNotFoundError:
        logger.info(f"File not found: {gridfile}")
        raise
    except KeyError as e:
        logger.info(f"Missing field in .mat file: {e}")
        raise

    return input_data


def init_initial_conditions(grid, io, time):
    """
    Sets the model's initial conditions at the start of the simulation.

    Parameters:
        grid (dict): Dictionary representing the grid, including fields like `gpsum`, `nl`, `max_subZ`, `split`, etc.
        io (dict): Dictionary with I/O settings (e.g. readbootfile, rebootdir, bootfilein, homedir).

    Returns:
        OUT (dict): Dictionary containing model outputs initialized with default or restart file values.
        IN (dict): Dictionary containing model inputs initialized with default values.
        OUTFILE (dict): Placeholder dictionary for output file management.
    """

    OUT = {}  # Dictionary to hold model output variables
    IN = {}  # Dictionary to hold model input variables

    gpsum = grid["gpsum"]
    nl = grid["nl"]

    ##########################################################
    # Initialize conditions from restart file or set manually
    ##########################################################
    if io.get("readbootfile", False):
        logger.info("EBFM: Initialize from restart file...")

        reboot_dir = io["rebootdir"]
        boot_filepath = f"{reboot_dir}/{io['bootfilein']}"

        # Open the NetCDF file
        with Dataset(boot_filepath, "r") as ncfile:
            # Iterate through all variables in the file
            for var_name in ncfile.variables:
                # Read the variable data
                var_data = ncfile.variables[var_name][:]
                # If a variable has no dimensions (scalar), convert it to a Python scalar
                if var_data.shape == ():  # Scalar variable
                    var_data = var_data.item()
                # Save the variable data to the dictionary
                OUT[var_name] = var_data

        OUT["timelastsnow"] = num2date(
            OUT["timelastsnow_netCDF"],
            units="days since 1970-01-01 00:00:00",
            calendar="gregorian",
        )

    else:
        logger.info("EBFM: Initialize from manually set conditions...")

        OUT["Tsurf"] = np.full((gpsum,), 273.15)  # Surface temperature (K)
        OUT["subT"] = np.full((gpsum, nl), 265.0)  # Vertical temperatures (K)
        OUT["subW"] = np.zeros((gpsum, nl))  # Vertical irreducible water content (kg)
        OUT["subS"] = np.zeros((gpsum, nl))  # Vertical slush water content (kg)
        OUT["subD"] = np.full((gpsum, nl), Ice.DENSITY)  # Vertical densities (kg m-3)
        OUT["subTmean"] = OUT["subT"]  # Annual mean vertical layer temperature (K)
        OUT["timelastsnow"] = np.full((gpsum,), time["ts"])  # Timestep of last snowfall (days)
        OUT["ys"] = np.full((gpsum,), 500.0)  # Annual snowfall (mm water equivalent)
        OUT["subZ"] = np.full((gpsum, nl), grid["max_subZ"])  # Vertical layer depths (m)
        OUT["alb_snow"] = np.full((gpsum,), FreshSnow.ALBEDO)  # Snow albedo
        OUT["snowmass"] = np.zeros((gpsum,))  # Snow mass (m water equivalent)

        if grid.get("doubledepth", False):  # Sets layer thicknesses when 'double depth' is active
            mask_indices = np.where(grid["mask"] == 1)
            split = grid["split"]
            for n, split_start in enumerate(split[:-1]):
                depth_value = (2.0**n) * grid["max_subZ"]
                OUT["subZ"][mask_indices[0], split_start : split[n + 1]] = depth_value

            final_depth_value = (2.0 ** len(split)) * grid["max_subZ"]
            OUT["subZ"][mask_indices[0], split[-1] :] = final_depth_value

    ######################################################
    # Declare non-initialized variables in `OUT`
    ######################################################
    OUT["smb_cumulative"] = np.zeros((gpsum,))  # Cumulative climatic mass balance (m w.e.)
    OUT["smb"] = np.zeros((gpsum,))  # Climatic mass balance (m w.e.)
    OUT["subK"] = np.zeros((gpsum, nl))  # Vertical conductivity (m2 s-1)
    OUT["subCeff"] = np.zeros((gpsum, nl))  # Vertical effective heat capacity (J m-3 K)
    OUT["subWvol"] = np.zeros((gpsum, nl))  # Vertical volumetric water content (fraction)
    OUT["surfH"] = np.zeros((gpsum,))  # Surface height (m)
    OUT["Dfreshsnow"] = np.zeros((gpsum,))  # Fresh snow density (kg m-3)
    OUT["tstar"] = np.zeros((gpsum,))  # Albedo decay timescale (days)
    OUT["runoff_irr_deep_mean"] = np.zeros((gpsum,))  # Runoff irreducible water (m w.e.)
    OUT["albedo"] = np.full((gpsum,), Ice.ALBEDO)  # Albedo (fraction)

    ######################################################
    # Initialize `IN` (model input variables)
    ######################################################
    IN["T"] = np.zeros((gpsum,))  # Air temperature (K)
    IN["P"] = np.zeros((gpsum,))  # Precipitation (m w.e.)
    IN["snow"] = np.zeros((gpsum,))  # Snow precipitation (m w.e.)
    IN["rain"] = np.zeros((gpsum,))  # Rain precipitation (m w.e.)
    IN["yearsnow"] = np.zeros((gpsum, nl))  # Annual snow precipitation (m w.e.)
    IN["logyearsnow"] = np.zeros((gpsum, nl))  # Log annual snow precipitation (m w.e.)
    IN["C"] = np.zeros((gpsum,))  # Cloud cover fraction
    IN["WS"] = np.zeros((gpsum,))  # Wind speed (m s-1)
    IN["RH"] = np.zeros((gpsum,))  # Relative humidity (fraction)
    IN["q"] = np.zeros((gpsum,))  # Specific humidity (g kg-1)
    IN["VP"] = np.zeros((gpsum,))  # Vapor pressure (Pa)
    IN["Dair"] = np.zeros((gpsum,))  # Air density (kg m-3)
    IN["Pres"] = np.zeros((gpsum,))  # Air pressure (Pa)
    IN["SWin"] = np.zeros((gpsum,))  # Incoming SW radiation (W m-2)
    IN["LWin"] = np.zeros((gpsum,))  # Incoming LW radiation (W m-2)

    OUTFILE = {}  # Placeholder for output to be saved to files

    return OUT, IN, OUTFILE


# Example usage: Initialize constants and print
if __name__ == "__main__":
    constants = init_constants()

    # Print constants to verify
    for key, value in constants.items():
        print(f"{key}: {value}")
