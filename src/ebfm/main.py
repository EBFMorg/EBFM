# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ebfm.core
import ebfm.core.comm

from ebfm.core import (
    INIT,
    LOOP_general_functions,
    LOOP_climate_forcing,
    LOOP_EBM,
    LOOP_SNOW,
    LOOP_mass_balance,
)
from ebfm.core import LOOP_write_to_file, FINAL_create_restart_file
from ebfm.core.grid import GridInputType
from ebfm.core.config import CouplingConfig, GridConfig, TimeConfig
from ebfm.core.logger import Logger, setup_logging, log_levels_map, getLogger
from ebfm.core.cli import (
    extract_active_coupling_features,
    parse_cli_args,
    resolve_numba_threads,
    validate_shading_coupling_compat,
)

import ebfm.coupling

# logger for this module
logger: Logger
# dedicated logger for diagnostic output (--print-diagnostics)
diagnostics_logger = getLogger("ebfm.diagnostics")


# Arrays saved by --dump-reference
_REFERENCE_KEYS = [
    "smb",
    "smb_cumulative",
    "Tsurf",
    "subT",
    "subD",
    "subZ",
    "subW",
    "subS",
    "surfH",
    "subTmean",
    "runoff_irr",
    "Dens_destr_metam",
    "Dens_overb_pres",
    "Dens_drift",
    # reboot / restart state
    "snowmass",
    "ys",
    "timelastsnow_netCDF",
    "alb_snow",
]


def _format_stats(arr: np.ndarray) -> str:
    """Format min/max/mean statistics of an array for diagnostic log output."""
    return f"min={arr.min():.4e}  max={arr.max():.4e}  mean={arr.mean():.4e}"


def dump_reference(logger, OUT, filepath: str):
    """Save key output arrays to a .npz file for later comparison."""
    import numpy as np

    data = {k: OUT[k] for k in _REFERENCE_KEYS if k in OUT}
    missing = [k for k in _REFERENCE_KEYS if k not in OUT]
    if missing:
        logger.warning(f"[DUMP] Keys not found in OUT and will be skipped: {missing}")
    np.savez(filepath, **data)
    logger.info(f"[DUMP] Reference snapshot saved to '{filepath}' (keys: {list(data.keys())})")


def print_diagnostics(grid, OUT, t):
    """Log key diagnostic values each timestep for performance and correctness analysis."""

    gpsum = grid.get("gpsum", "N/A")
    has_shading = grid.get("has_shading", False)
    smb = OUT.get("smb")
    smb_cum = OUT.get("smb_cumulative")

    diagnostics_logger.info(f"[t={t + 1}] gpsum={gpsum}, shading={'on' if has_shading else 'off'}")
    if smb is not None:
        diagnostics_logger.info(f"[t={t + 1}] smb:            {_format_stats(smb)}")
    if smb_cum is not None:
        diagnostics_logger.info(f"[t={t + 1}] smb_cumulative: {_format_stats(smb_cum)}")


def _main_impl():
    args = parse_cli_args()

    # Bootstrap logging before communicator splitting so early diagnostics are available.
    setup_logging(
        stdout_log_level=log_levels_map[args.log_level_console],
        file=args.log_file,
        reset_handlers=True,
    )
    logger = getLogger(__name__)
    logger.debug("Bootstrap logging initialized with MPI.COMM_WORLD.")

    time_config = TimeConfig(args)

    active_coupling_features = extract_active_coupling_features(args)
    coupling_config = CouplingConfig(args, time_config)
    ebfm.coupling.check_coupling_requirements(coupling_config, active_coupling_features)

    if ebfm.core.comm.mpi_available:
        from ebfm.core.comm import mpi as comm_mpi

        ebfm_comm = comm_mpi.do_comm_splitting(args.local_group_label, coupling_config)
    else:
        ebfm_comm = ebfm.core.comm.defaultComm

    # Reconfigure logging for (now available) EBFM communicator.
    setup_logging(
        stdout_log_level=log_levels_map[args.log_level_console],
        file=args.log_file,
        comm=ebfm_comm,
        reset_handlers=True,
    )

    if not hasattr(args, "use_part"):
        # If not provided via command line option --use-part, set to rank + 1 (assuming partition IDs start at 1).
        args.use_part = ebfm_comm.rank + 1

    logger = getLogger(__name__)
    logger.info(f"Starting EBFM version {ebfm.core.get_version()}...")

    # Numba is opt-in: only activate when --with-numba is explicitly passed.
    if args.with_numba:
        from ebfm.core.compute_backend import init_numba

        n_threads = resolve_numba_threads(args, ebfm_comm, logger)
        init_numba(n_threads)
    else:
        logger.info(
            "Pass --with-numba to enable the parallel Numba kernels " "(requires: pip install 'ebfm[performance]')."
        )

    logger.info("Done parsing command line arguments.")
    logger.debug("Parsed the following command line arguments:")
    for arg, val in vars(args).items():
        logger.debug(f"  {arg}: {val}")

    logger.debug("Reading configuration and checking for consistency.")

    # TODO consider introducing an ebfm_adapter_config.yaml to be parsed alternatively/additionally to command line args
    grid_config = GridConfig(args)

    # Ensure shading routine is only used in uncoupled runs.
    # See https://github.com/EBFMorg/EBFM/issues/11 for details.
    validate_shading_coupling_compat(grid_config, coupling_config)

    logger.debug("Successfully completed consistency checks.")

    # Model setup & initialization
    time = time_config.to_dict()
    grid, io, phys = INIT.init_config(time_config, grid_config, args.restart_dir, args.restart_init)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        logger.info(f"NumPy random seed fixed to {args.random_seed} for reproducibility.")

    C = INIT.init_constants()
    grid = INIT.init_grid(grid, io, grid_config)

    OUT, IN, OUTFILE = INIT.init_initial_conditions(C, grid, io, time, init_with_restart_file=args.restart_init)

    coupler_cls = ebfm.coupling.select_coupler_class(coupling_config)
    coupler = coupler_cls(coupling_config=coupling_config)

    coupler.setup(grid, time_config)

    # Time-loop
    logger.info("Entering time loop...")
    for t in range(time["tn"]):
        # Print time to screen
        time["TCUR"] = LOOP_general_functions.print_time(t, time["ts"], time["dt"])

        logger.info(f'Time step {t + 1} of {time["tn"]} (dt = {time["dt"]} days)')

        # Read and prepare climate input
        if coupler.has_coupling_to("icon_atmo"):
            # Exchange data with ICON
            icon_atmo = coupler.get_component("icon_atmo")
            logger.info("Data exchange with ICON")
            logger.debug("Started...")
            data_to_icon = {
                "albedo": OUT["albedo"],
            }

            data_from_icon = icon_atmo.exchange(data_to_icon)

            logger.debug("Done.")
            logger.debug(f"Received the following data from ICON: {data_from_icon}")

            IN["P"] = data_from_icon["pr"]
            IN["snow"] = data_from_icon["pr_snow"]
            IN["SWin"] = data_from_icon["rsds"]
            IN["LWin"] = data_from_icon["rlds"]
            IN["C"] = data_from_icon["clt"]
            IN["WS"] = data_from_icon["sfcwind"]
            IN["T"] = data_from_icon["tas"]
            IN["rain"] = IN["P"] - IN["snow"]  # TODO: make this more flexible and configurable
            # Fallback to constants if fields are not coupled (error code set); must be arrays for mask indexing.
            _T0 = IN["T"] * 0.0

            if "huss" in data_from_icon:
                IN["q"] = data_from_icon["huss"]
            else:  # use fallback value
                IN["q"] = _T0

            if "sfcpres" in data_from_icon:
                IN["Pres"] = data_from_icon["sfcpres"]
            else:  # use fallback value
                IN["Pres"] = _T0 + 101500.0

        # Read/set meteorological forcing
        IN, OUT = LOOP_climate_forcing.main(C, grid, IN, t, time, OUT, coupler)

        # Run surface energy balance model
        OUT = LOOP_EBM.main(C, OUT, IN, time, grid, coupler)

        # Run snow & firn model
        OUT = LOOP_SNOW.main(C, OUT, IN, time["dt"], grid, phys)

        # Calculate surface mass balance
        OUT = LOOP_mass_balance.main(OUT, IN, C)

        if args.print_diagnostics:
            print_diagnostics(grid, OUT, t)

        if coupler.has_coupling_to("elmer_ice"):
            elmer_ice = coupler.get_component("elmer_ice")
            # Exchange data with Elmer
            logger.info("Data exchange with Elmer/Ice")
            logger.debug("Started...")

            data_to_elmer = {
                "smb": OUT["smb"],
                "T_ice": OUT["T_ice"],
                "runoff": OUT["runoff"],
            }
            data_from_elmer = elmer_ice.exchange(data_to_elmer)
            logger.debug("Done.")
            logger.debug(f"Received the following data from Elmer/Ice: {data_from_elmer}")

            IN["surface_elevation"] = data_from_elmer["surface_elevation"]
            OUT["surface_elevation"] = IN["surface_elevation"]
            OUT["x"] = grid["x"]
            OUT["y"] = grid["y"]
            if coupler.has_coupling_to("icon_atmo"):
                grid["z"] = IN["surface_elevation"][0].ravel()
            # TODO add gradient field later
            # IN['dhdx'] = data_from_elmer('dhdx')
            # IN['dhdy'] = data_from_elmer('dhdy')
        else:
            # Needed by FINAL_create_restart_file.main(OUT, io)
            OUT["x"] = grid["x"]
            OUT["y"] = grid["y"]
            OUT["surface_elevation"] = grid["z"]

        # Write output to files (only in uncoupled run and for unpartitioned grid)
        # TODO: should be supported for all cases to avoid case distinction here
        if not grid["is_partitioned"] and isinstance(coupler, ebfm.coupling.DummyCoupler):
            if grid_config.grid_type is GridInputType.MATLAB:
                io, OUTFILE = LOOP_write_to_file.main(OUTFILE, io, OUT, grid, t, time)
            else:
                logger.warning("Skipping writing output to file for Elmer input grids.")
        elif grid["is_partitioned"] or not isinstance(coupler, ebfm.coupling.DummyCoupler):
            logger.warning("Skipping writing output to file for coupled or partitioned runs.")
        else:
            logger.error("Unhandled case in output writing.")
            raise Exception("Unhandled case in output writing.")

    # Write restart file
    # TODO: should be supported for all cases to avoid case distinction here
    if not grid["is_partitioned"]:
        FINAL_create_restart_file.main(OUT, io, args.restart_dir)
    else:
        logger.warning("Skipping writing of restart file for coupled and/or partitioned runs.")

    logger.info("Time loop completed.")

    if args.dump_reference:
        dump_reference(logger, OUT, args.dump_reference)

    coupler.finalize()

    logger.info("Closing down EBFM.")


def main():
    try:
        _main_impl()
    except Exception as e:
        logger = getLogger(__name__)
        logger.exception("Fatal Error in EBFM")

        if ebfm.core.comm.mpi_available:
            from ebfm.core.comm import mpi as comm_mpi

            comm_mpi.abort()

        raise e


# Entry point for script execution
if __name__ == "__main__":
    main()
