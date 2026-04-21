# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from datetime import datetime
import argparse
from enum import Enum
import numpy as np

import ebfm.core
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
from ebfm.core.config import CouplingConfig, GridConfig, TimeConfig, FieldValidationLevel
from ebfm.core.logger import Logger, setup_logging, log_levels_map, getLogger

import ebfm.coupling

from mpi4py import MPI

# logger for this module
logger: Logger


class CliDefaults(Enum):
    START_TIME = datetime(1979, 1, 1, 0, 0)
    END_TIME = datetime(1979, 1, 2, 0, 0)
    FIELD_VALIDATION_LEVEL = FieldValidationLevel.FATAL
    TIME_STEP_SIZE_IN_DAYS = 0.125  # = 0.125 days = 3 hours
    LOG_LEVEL_CONSOLE = "INFO"
    COMPONENT_NAME = "ebfm"

    @classmethod
    def default_time_step_size_in_hours(cls) -> float:
        return cls.TIME_STEP_SIZE_IN_DAYS.value * 24


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

    coupling_group.add_argument(
        "--field-validation-level",
        type=str,
        choices={level.value for level in FieldValidationLevel},
        default=CliDefaults.FIELD_VALIDATION_LEVEL.value,
        help="Level of validation for field exchange type checks. "
        "'FATAL': raise exception on mismatch (default), "
        "'WARNING': log warning on mismatch, "
        "'SILENT': only log at debug level on mismatch.",
    )

    coupling_group.add_argument(
        "--fake-coupling",
        action="store_true",
        help="Use FakeCoupler to provide synthetic data for coupled fields without requiring YAC or actual coupled "
        "models. Useful for testing the coupling infrastructure.",
    )

    coupling_group.add_argument(
        "--component-name",
        type=str,
        default=CliDefaults.COMPONENT_NAME.value,
        help="Identifier for this EBFM instance used by the coupler.",
    )


def extract_active_coupling_features(args: argparse.Namespace) -> list[str]:
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


def dump_reference(logger, OUT, filepath: str):
    """Save key output arrays to a .npz file for later comparison."""
    import numpy as np

    data = {k: OUT[k] for k in _REFERENCE_KEYS if k in OUT}
    missing = [k for k in _REFERENCE_KEYS if k not in OUT]
    if missing:
        logger.warning(f"[DUMP] Keys not found in OUT and will be skipped: {missing}")
    np.savez(filepath, **data)
    logger.info(f"[DUMP] Reference snapshot saved to '{filepath}' (keys: {list(data.keys())})")


def print_diagnostics(logger, grid, OUT, t):
    """Log key diagnostic values each timestep for performance and correctness analysis."""

    gpsum = grid.get("gpsum", "N/A")
    has_shading = grid.get("has_shading", False)
    smb = OUT.get("smb")
    smb_cum = OUT.get("smb_cumulative")

    logger.info(f"[DIAG t={t + 1}] gpsum={gpsum}, shading={'on' if has_shading else 'off'}")
    if smb is not None:
        logger.info(f"[DIAG t={t + 1}] smb:            min={smb.min():.4e}  max={smb.max():.4e}  mean={smb.mean():.4e}")
    if smb_cum is not None:
        logger.info(
            f"[DIAG t={t + 1}] smb_cumulative: "
            f"min={smb_cum.min():.4e}  "
            f"max={smb_cum.max():.4e}  "
            f"mean={smb_cum.mean():.4e}"
        )


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
        "--shading",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable shading. Defaults to True for MATLAB meshes, False for all other mesh types.",
    )

    input_group.add_argument(
        "--netcdf-mesh-unstructured",
        type=Path,
        help="Path to the unstructured NetCDF mesh file. Optional if using --elmer-mesh."
        " If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    example_restart_file_name = INIT.create_restart_file_name(CliDefaults.START_TIME.value)

    input_group.add_argument(
        "--restart-dir",
        type=Path,
        help="Path to folder with restart files. If --restart-init there must be a restart file in "
        f" the folder must be named after given --start-time (e.g., '{example_restart_file_name}').",
    )

    input_group.add_argument(
        "--restart-init",
        action="store_true",
        help="Initialise from restart file in --restart-dir.",
    )

    input_group.add_argument(
        "--elmer-mesh-crs-epsg",
        type=int,
        choices={
            3413,  # EPSG code for NSIDC Sea Ice Polar Stereographic North (commonly used for Greenland)
            3031,  # EPSG code for NSIDC Sea Ice Polar Stereographic South (commonly used for Antarctica)
        },
        help="EPSG code of the input Elmer mesh coordinate reference system."
        " Used to convert mesh x/y coordinates to lon/lat."
        " Required when using --elmer-mesh.",
    )

    time_group = parser.add_argument_group("time configuration")

    time_group.add_argument(
        "--start-time",
        type=str,
        help=f"Start time of the simulation in format '{TimeConfig.input_time_format_display}' "
        "(i.e., time at the beginning of the first time step)",
        default=CliDefaults.START_TIME.value.strftime(TimeConfig.input_time_format),
    )

    time_group.add_argument(
        "--end-time",
        type=str,
        help=f"End time of the simulation in format '{TimeConfig.input_time_format_display}' "
        "(i.e., time at the end of the last time step)",
        default=CliDefaults.END_TIME.value.strftime(TimeConfig.input_time_format),
    )

    time_group.add_argument(
        "--time-step",
        type=float,
        help=f"Time step of the simulation in days, e.g., {CliDefaults.TIME_STEP_SIZE_IN_DAYS.value} for "
        f"{CliDefaults.default_time_step_size_in_hours()} hours. Note: The difference between --end-time and "
        "--start-time must be divisible by --time-step.",
        default=CliDefaults.TIME_STEP_SIZE_IN_DAYS.value,
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

    logger_group = parser.add_argument_group("logging configuration")

    logger_group.add_argument(
        "--log-level-console",
        type=str,
        choices=list(log_levels_map.keys()),
        default=CliDefaults.LOG_LEVEL_CONSOLE.value,
        help="Log level for console output for all MPI ranks (unless overridden by custom settings in utils.py).",
    )

    logger_group.add_argument(
        "--log-file",
        type=Path,
        help="If provided, log output will be written to the specified file (one file per MPI rank).",
    )

    diag_group = parser.add_argument_group("diagnostics, reference snapshots, random seed")

    diag_group.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help=(
            "Log diagnostic info each timestep via the INFO logger: "
            "gpsum, shading status, and smb / smb_cumulative statistics."
        ),
    )

    diag_group.add_argument(
        "--dump-reference",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "After the final timestep, save key output arrays "
            "(see defined `_REFERENCE_KEYS`) to FILE as a NumPy .npz archive. "
            "Use tools/compare_snapshots.py to diff two such files."
        ),
    )

    diag_group.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Fix the NumPy random seed before the time loop. Required for reproducible "
            "results when using random climate forcing (e.g. set_random_weather_data), "
            "so that --dump-reference snapshots from two identical runs can be compared."
        ),
    )

    performance_group = parser.add_argument_group("Performance and Numba configuration")

    performance_group.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of Numba threads to use per MPI rank. "
            "Requires numba; install via 'pip install ebfm[performance]'. "
        ),
    )

    performance_group.add_argument(
        "--with-numba",
        action="store_true",
        default=False,
        help=("Enable the parallel Numba kernel. " "Requires numba; install via: pip install 'ebfm[performance]'. "),
    )

    # Add args for features requiring 'import coupling'
    add_coupling_arguments(parser)

    args = parser.parse_args()

    if args.version:
        ebfm.core.print_version_and_exit()

    # Validate that --elmer-mesh-crs-epsg is provided when using --elmer-mesh
    if args.elmer_mesh and args.elmer_mesh_crs_epsg is None:
        parser.error("--elmer-mesh-crs-epsg is required when using --elmer-mesh")

    active_coupling_features = extract_active_coupling_features(args)
    coupling_config = CouplingConfig(args)
    ebfm.coupling.check_coupling_requirements(coupling_config, active_coupling_features)

    # TODO: replace MPI.COMM_WORLD with communicator from ebfm; either from couplers comm splitting or default comm
    setup_logging(
        stdout_log_level=log_levels_map[args.log_level_console],
        file=args.log_file,
        comm=MPI.COMM_WORLD,
    )

    logger = getLogger(__name__)
    logger.info(f"Starting EBFM version {ebfm.core.get_version()}...")

    # Numba is opt-in: only activate when --with-numba is explicitly passed.
    # set_num_threads() must be called before any kernel runs.
    if args.with_numba:
        if not LOOP_SNOW._NUMBA_AVAILABLE:
            parser.error("--with-numba: numba is not installed. " "Run: pip install 'ebfm[performance]'")

        import multiprocessing as _mp
        import numba as _numba_mod

        cpu_count = _mp.cpu_count()
        mpi_size = MPI.COMM_WORLD.size
        suggested_max_threads = max(1, cpu_count // mpi_size)

        # Default: 1 thread unless user explicitly sets --numba-threads
        n_threads = args.numba_threads if args.numba_threads is not None else 1

        if args.numba_threads is None:
            logger.info(
                f"[NUMBA] --with-numba enabled. Using default threads per rank: {n_threads}. "
                f"To increase, pass --numba-threads N (suggested max: {suggested_max_threads}; "
                f"cpu_count={cpu_count}, MPI world size={mpi_size})."
            )
        else:
            if n_threads < 1:
                parser.error("--numba-threads must be a >= 1")
            if n_threads > suggested_max_threads:
                parser.error(
                    f"[NUMBA] --numba-threads={n_threads} exceeds suggested max {suggested_max_threads} "
                    f"(cpu_count={cpu_count}, MPI world size={mpi_size}). "
                    f"--numba-threads must be between 1 and {suggested_max_threads} on this system. "
                )

            logger.info(
                f"[NUMBA] --with-numba enabled. Threads per rank: {n_threads} "
                f"(suggested max {suggested_max_threads}; cpu_count={cpu_count}, MPI world size={mpi_size})."
            )

        _numba_mod.set_num_threads(n_threads)
        LOOP_SNOW._USE_NUMBA = True

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

    # Ensure shading routine is only used in uncoupled runs
    # see https://github.com/EBFMorg/EBFM/issues/11 for details.
    if grid_config.use_shading and coupling_config.defines_coupling():
        parser.error(
            "Shading routine not implemented for coupled runs. "
            "Please deactivate shading via --no-shading or deactivate coupling."
        )

    time_config = TimeConfig(args)

    logger.debug("Successfully completed consistency checks.")

    # Model setup & initialization
    time = time_config.to_dict()
    grid, io, phys = INIT.init_config(time_config, grid_config, args.restart_dir, args.restart_init)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        logger.info(f"[DIAG] NumPy random seed fixed to {args.random_seed} for reproducibility.")

    C = INIT.init_constants()
    grid = INIT.init_grid(grid, io, grid_config)

    OUT, IN, OUTFILE = INIT.init_initial_conditions(C, grid, io, time, init_with_restart_file=args.restart_init)

    coupler_cls = ebfm.coupling.select_coupler_class(coupling_config)
    coupler = coupler_cls(coupling_config=coupling_config)

    coupler.setup(grid, time)

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

        logger.info("EBFM main calculations")

        IN, OUT = LOOP_climate_forcing.main(C, grid, IN, t, time, OUT, coupler)

        # Run surface energy balance model
        OUT = LOOP_EBM.main(C, OUT, IN, time, grid, coupler)

        # Run snow & firn model
        OUT = LOOP_SNOW.main(C, OUT, IN, time["dt"], grid, phys)

        # Calculate surface mass balance
        OUT = LOOP_mass_balance.main(OUT, IN, C)

        if args.diagnostics:
            print_diagnostics(logger, grid, OUT, t)

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
            logger.debug("Received the following data from Elmer/Ice:", data_from_elmer)

            IN["h"] = data_from_elmer["h"]
            OUT["h"] = IN["h"]
            OUT["x"] = grid["x"]
            OUT["y"] = grid["y"]
            if coupler.has_coupling_to("icon_atmo"):
                grid["z"] = IN["h"][0].ravel()
            # TODO add gradient field later
            # IN['dhdx'] = data_from_elmer('dhdx')
            # IN['dhdy'] = data_from_elmer('dhdy')
        else:
            # Needed by FINAL_create_restart_file.main(OUT, io)
            OUT["x"] = grid["x"]
            OUT["y"] = grid["y"]
            OUT["h"] = grid["z"]

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


# Entry point for script execution
if __name__ == "__main__":
    main()
