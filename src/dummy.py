# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import coupling
from pathlib import Path
import argparse

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

from mpi4py import MPI
from utils import setup_logging
import logging

log_levels = {
    "file": logging.DEBUG,  # log level for logging to file
    0: logging.INFO,  # log level for rank 0
    # 1: logging.DEBUG,  # to log other ranks to console define log level here
}
setup_logging(log_levels=log_levels)

# logger for this module
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting EBFM...")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--couple-to-elmer-ice",
        action="store_true",
        help="Enable coupling with Elmer/Ice models via YAC",
    )

    parser.add_argument(
        "--couple-to-icon-atmo",
        action="store_true",
        help="Enable coupling with ICON via YAC",
    )

    parser.add_argument(
        "--coupler-config",
        type=Path,
        help="Path to the coupling configuration file (YAC coupler_config.yaml).",
    )

    parser.add_argument(
        "--elmer-mesh",
        type=Path,
        help="Path to the Elmer mesh file. Either --elmer-mesh or --matlab-mesh is required.",
    )

    parser.add_argument(
        "--matlab-mesh",
        type=Path,
        help="Path to the MATLAB mesh file. Either --elmer-mesh or --matlab-mesh is required.",
    )

    parser.add_argument(
        "--netcdf-mesh",
        type=Path,
        help="Path to the NetCDF mesh file. Optional if using --elmer-mesh."
        " If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    parser.add_argument(
        "--is-partitioned-elmer-mesh",
        action="store_true",
        help="Indicate if the provided Elmer mesh is partitioned for parallel runs.",
    )

    parser.add_argument(
        "--use-part",
        type=int,
        default=MPI.COMM_WORLD.rank + 1,
        help="If using a partitioned Elmer mesh, allows to specify which partition ID to use for this run. "
        "If not provided, the MPI rank + 1 will be used as partition ID.",
    )

    args = parser.parse_args()

    logger.info("Done parsing command line arguments.")
    logger.debug("Parsed the following command line arguments:")
    for arg, val in vars(args).items():
        logger.debug(f"  {arg}: {val}")

    # Model setup & initialization
    grid, time2, io, phys = INIT.init_config(args)
    C = INIT.init_constants()
    grid = INIT.init_grid(grid, io, args)

    OUT, IN, OUTFILE = INIT.init_initial_conditions(C, grid, io, time2)

    if args.couple_to_icon_atmo or args.couple_to_elmer_ice:
        # TODO: introduce minimal stub implementation
        # TODO consider introducing an ebfm_adapter_config.yaml
        coupler = coupling.init(
            coupler_config=args.coupler_config,
            ebfm_coupling_config=Path("dummies") / "EBFM" / "ebfm-config.yaml",
            couple_with_icon_atmo=args.couple_to_icon_atmo,
            couple_with_elmer_ice=args.couple_to_elmer_ice,
        )
        coupling.setup(coupler, grid["mesh"], time2)
    else:
        coupler = coupling.NoCoupler("ebfm")

    # Time-loop
    logger.info("Entering time loop...")
    for t in range(1, time2["tn"] + 1):
        # Print time to screen
        time2 = LOOP_general_functions.print_time(t, time2)

        logger.info(f'Time step {t} of {time2["tn"]} (dt = {time2["dt"]} days)')

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
                data_from_icon["pr"] * time2["dt"] * C["dayseconds"] * 1e-3
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

        IN, OUT = LOOP_climate_forcing.main(C, grid, IN, t, time2, OUT, coupler)

        # Run surface energy balance model
        OUT = LOOP_EBM.main(C, OUT, IN, time2, grid, coupler)

        # Run snow & firn model
        OUT = LOOP_SNOW.main(C, OUT, IN, time2["dt"], grid, phys)

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
            assert (
                grid["input_type"] is GridInputType.MATLAB
            ), "Output writing currently only implemented for MATLAB input grids."
            io, OUTFILE = LOOP_write_to_file.main(OUTFILE, io, OUT, grid, t, time2, C)
            pass
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
