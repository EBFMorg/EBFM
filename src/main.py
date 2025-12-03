# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import argparse

import ebfm
from ebfm import (
    INIT,
    LOOP_general_functions,
    LOOP_climate_forcing,
    LOOP_EBM,
    LOOP_SNOW,
    LOOP_mass_balance,
)
from ebfm import LOOP_write_to_file, FINAL_create_restart_file
from ebfm.grid import GridInputType
from ebfm.config import CouplingConfig, GridConfig, TimeConfig

from mpi4py import MPI
from utils import setup_logging
import logging

from typing import List

try:
    import coupling  # noqa: E402

    coupling_supported = True
except ImportError as e:
    coupling_supported = False
    coupling_import_error = e

log_levels = {
    "file": logging.DEBUG,  # log level for logging to file
    0: logging.INFO,  # log level for rank 0
    # 1: logging.DEBUG,  # to log other ranks to console define log level here
}
setup_logging(log_levels=log_levels)

# logger for this module
logger = logging.getLogger(__name__)


def add_coupling_arguments(parser: argparse.ArgumentParser):
    """
    Add command line arguments related to coupling with other models via YAC.

    @param[in] parser the argument parser to add the coupling arguments to.
    """

    # Note: If you add arguments to this function, also update check_coupling_features.

    coupling_group = parser.add_argument_group("coupling (requires YAC)")

    coupling_group.add_argument(
        "--couple-to-elmer-ice",
        action="store_true",
        help="Enable coupling with Elmer/Ice models via YAC",
    )

    coupling_group.add_argument(
        "--couple-to-icon-atmo",
        action="store_true",
        help="Enable coupling with ICON via YAC",
    )

    coupling_group.add_argument(
        "--coupler-config",
        type=Path,
        help="Path to the coupling configuration file (YAC coupler_config.yaml).",
    )


def extract_active_coupling_features(args: argparse.Namespace) -> List[str]:
    """
    Determine if coupling is required based on the provided command line arguments.

    @param[in] args the parsed command line arguments.

    @return a list of argument names that indicate coupling is required.
    """

    active_coupling_args = []

    if args.couple_to_elmer_ice:
        active_coupling_args.append("--couple-to-elmer-ice")

    if args.couple_to_icon_atmo:
        active_coupling_args.append("--couple-to-icon-atmo")

    if args.coupler_config:
        active_coupling_args.append("--coupler-config")

    return active_coupling_args


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the EBFM version and exit.",
    )

    input_group = parser.add_argument_group("input mesh types")

    input_group.add_argument(
        "--elmer-mesh",
        type=Path,
        help="Path to the Elmer mesh file. Either --elmer-mesh or --matlab-mesh is required.",
    )

    input_group.add_argument(
        "--matlab-mesh",
        type=Path,
        help="Path to the MATLAB mesh file. Either --elmer-mesh or --matlab-mesh is required.",
    )

    input_group.add_argument(
        "--netcdf-mesh",
        type=Path,
        help="Path to the NetCDF mesh file. Optional if using --elmer-mesh."
        " If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    input_group.add_argument(
        "--netcdf-mesh-unstructured",
        type=Path,
        help="Path to the unstructured NetCDF mesh file. Optional if using --elmer-mesh."
        " If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    time_group = parser.add_argument_group("time configuration")

    time_group.add_argument(
        "--start-time",
        type=str,
        help="Start time of the simulation in format 'DD-Mon-YYYY HH:MM'",
        default="1-Jan-1979 00:00",
    )

    time_group.add_argument(
        "--end-time",
        type=str,
        help="End time of the simulation in format 'DD-Mon-YYYY HH:MM'",
        default="2-Jan-1979 00:00",
    )

    time_group.add_argument(
        "--time-step",
        type=float,
        help="Time step of the simulation in days, e.g., 0.125 for 3 hours.",
        default=0.125,
    )

    parallel_group = parser.add_argument_group("parallel runs and distributed meshes")

    parallel_group.add_argument(
        "--is-partitioned-elmer-mesh",
        action="store_true",
        help="Indicate if the provided Elmer mesh is partitioned for parallel runs.",
    )

    parallel_group.add_argument(
        "--use-part",
        type=int,
        default=MPI.COMM_WORLD.rank + 1,
        help="If using a partitioned Elmer mesh, allows to specify which partition ID to use for this run. "
        "If not provided, the MPI rank + 1 will be used as partition ID.",
    )

    # Add args for features requiring 'import coupling'
    add_coupling_arguments(parser)

    args = parser.parse_args()

    if args.version:
        ebfm.print_version_and_exit()

    has_active_coupling_features = extract_active_coupling_features(args)
    if has_active_coupling_features and not coupling_supported:
        raise RuntimeError(
            f"""
Coupling requested via command line argument(s) {has_active_coupling_features}, but the 'coupling' module could not be
imported due to the following error:

{coupling_import_error}

Hint: If you are missing 'yac', please install YAC and the python bindings as described under
https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html"
"""
        )
    else:
        from coupler import NoCoupler

    logger.info(f"Starting EBFM version {ebfm.get_version()}...")

    logger.info("Done parsing command line arguments.")
    logger.debug("Parsed the following command line arguments:")
    for arg, val in vars(args).items():
        logger.debug(f"  {arg}: {val}")

    logger.debug("Reading configuration and checking for consistency.")

    # TODO consider introducing an ebfm_adapter_config.yaml to be parsed alternatively/additionally to command line args
    coupling_config = CouplingConfig(args, component_name="ebfm")
    grid_config = GridConfig(args)
    time_config = TimeConfig(args)

    logger.debug("Successfully completed consistency checks.")

    # Model setup & initialization
    grid, io, phys = INIT.init_config()
    time = time_config.to_dict()

    C = INIT.init_constants()
    grid = INIT.init_grid(grid, io, grid_config)

    # Ensure shading routine is only used in uncoupled runs on unpartitioned MATLAB grids;
    # see https://github.com/EBFMorg/EBFM/issues/11 for details.
    if grid["has_shading"]:
        assert grid_config.is_partitioned is False, "Shading routine only implemented for unpartitioned grids."
        assert grid_config.grid_type is GridInputType.MATLAB, "Shading routine only implemented for MATLAB input grids."
        assert coupling_config.defines_coupling() is False, "Shading routine not implemented for coupled runs."

    OUT, IN, OUTFILE = INIT.init_initial_conditions(C, grid, io, time)

    if coupling_config.defines_coupling():
        # TODO: introduce minimal stub implementation
        # TODO consider introducing an ebfm_adapter_config.yaml
        coupler = coupling.init(coupling_config=coupling_config)
        coupling.setup(coupler, grid["mesh"], time)
    else:
        coupler = NoCoupler(component_name=coupling_config.component_name)

    # Time-loop
    logger.info("Entering time loop...")
    for t in range(1, time["tn"] + 1):
        # Print time to screen
        time["TCUR"] = LOOP_general_functions.print_time(t, time["ts"], time["dt"])

        logger.info(f'Time step {t} of {time["tn"]} (dt = {time["dt"]} days)')

        # Read and prepare climate input
        if coupler and coupler.couple_to_icon_atmo:
            # Exchange data with ICON
            logger.info("Data exchange with ICON")
            logger.debug("Started...")
            data_to_icon = {
                "albedo": OUT["albedo"],
            }

            data_from_icon = coupler.exchange_icon_atmo(data_to_icon)

            logger.debug("Done.")
            logger.debug("Received the following data from ICON:", data_from_icon)

            IN["P"] = (
                data_from_icon["pr"] * time["dt"] * C["dayseconds"] * 1e-3
            )  # convert units from kg m-2 s-1 to m w.e.
            IN["snow"] = data_from_icon["pr_snow"]
            IN["SWin"] = data_from_icon["rsds"]
            IN["LWin"] = data_from_icon["rlds"]
            IN["C"] = data_from_icon["clt"]
            IN["WS"] = data_from_icon["sfcwind"]
            IN["T"] = data_from_icon["tas"]
            IN["rain"] = IN["P"] - IN["snow"]  # TODO: make this more flexible and configurable
            IN["q"][:] = 0  # TODO: Read q from ICON instead and convert to RH
            IN["Pres"][:] = 101500  # TODO: Read Pres from ICON instead

        IN, OUT = LOOP_climate_forcing.main(C, grid, IN, t, time, OUT, coupler)

        # Run surface energy balance model
        OUT = LOOP_EBM.main(C, OUT, IN, time, grid, coupler)

        # Run snow & firn model
        OUT = LOOP_SNOW.main(C, OUT, IN, time["dt"], grid, phys)

        # Calculate surface mass balance
        OUT = LOOP_mass_balance.main(OUT, IN, C)

        if coupler.couple_to_elmer_ice:
            # Exchange data with Elmer
            logger.info("Data exchange with Elmer/Ice")
            logger.debug("Started...")

            data_to_elmer = {
                "smb": OUT["smb"],
                "T_ice": OUT["T_ice"],
                "runoff": OUT["runoff"],
            }
            data_from_elmer = coupler.exchange_elmer_ice(data_to_elmer)
            logger.debug("Done.")
            logger.debug("Received the following data from Elmer/Ice:", data_from_elmer)

            IN["h"] = data_from_elmer["h"]
            grid["z"] = IN["h"][0].ravel()
            # TODO add gradient field later
            # IN['dhdx'] = data_from_elmer('dhdx')
            # IN['dhdy'] = data_from_elmer('dhdy')

        # Write output to files (only in uncoupled run and for unpartitioned grid)
        if not grid["is_partitioned"] and not coupler.has_coupling:
            if grid_config.grid_type is GridInputType.MATLAB:
                io, OUTFILE = LOOP_write_to_file.main(OUTFILE, io, OUT, grid, t, time, C)
            else:
                logger.warning("Skipping writing output to file for Elmer input grids.")
        elif grid["is_partitioned"] or coupler.has_coupling:
            logger.warning("Skipping writing output to file for coupled or partitioned runs.")
        else:
            logger.error("Unhandled case in output writing.")
            raise Exception("Unhandled case in output writing.")

    # Write restart file
    if not grid["is_partitioned"] and not coupler.has_coupling:
        FINAL_create_restart_file.main(OUT, io)

    logger.info("Time loop completed.")

    if coupler.has_coupling:
        coupling.finalize(coupler)

    logger.info("Closing down EBFM.")


# Entry point for script execution
if __name__ == "__main__":
    main()
