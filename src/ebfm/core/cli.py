# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

"""Command-line interface for EBFM.

This module owns the full argument parser, all user-facing option compatibility checks, and the
default values for CLI arguments. Config classes (GridConfig, TimeConfig, …) then use assertions
to enforce internal invariants independently of the CLI layer.
"""

import argparse
from argparse import ArgumentParser, Namespace
from datetime import datetime
from enum import Enum
from pathlib import Path

import isodate

import ebfm.core
from ebfm.core import INIT
from ebfm.core.comm import mpi_available
from ebfm.core.config import Calendar, FieldValidationLevel, GridConfig, DEFAULT_TZ
from ebfm.core.grid import GridInputType
from ebfm.core.logger import log_levels_map

# Module-level parser reference, set by parse_cli_args().
# Allows post-parse validation functions to call parser.error() without
# requiring callers outside this module to hold a reference to the parser.
_parser: ArgumentParser | None = None


def _cli_error(msg: str) -> None:
    """Call parser.error() on the module-level parser. Must be called after parse_cli_args()."""
    assert _parser is not None, "_cli_error() called before parse_cli_args()"
    _parser.error(msg)


class CliDefaults(Enum):
    START_TIME = datetime(1979, 1, 1, 0, 0, tzinfo=DEFAULT_TZ)
    END_TIME = datetime(1979, 1, 2, 0, 0, tzinfo=DEFAULT_TZ)
    CALENDAR = Calendar.PROLEPTIC_GREGORIAN.value
    FIELD_VALIDATION_LEVEL = FieldValidationLevel.FATAL.value
    TIME_STEP_SIZE = "PT3H"  # = 3 hours
    LOG_LEVEL_CONSOLE = "INFO"
    COMPONENT_NAME = "ebfm"

    @classmethod
    def default_time_step_size_in_hours(cls) -> float:
        return isodate.parse_duration(cls.TIME_STEP_SIZE.value).total_seconds() / 3600.0


def add_coupling_arguments(parser: ArgumentParser) -> None:
    """Add command line arguments related to coupling with other models via YAC.

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


def extract_active_coupling_features(args: Namespace) -> list[str]:
    """Determine if coupling is required based on the provided command line arguments.

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


def parse_cli_args(args: list[str] | None = None) -> Namespace:
    """Build the argument parser, parse args, and validate option compatibility.

    Handles --version early exit before required arguments are enforced.

    All user-facing compatibility checks (e.g. conflicting or dependent options) are performed
    here via parser.error() so users see well-formatted, actionable error messages.

    The parser is stored in this module's _parser variable so that post-parse validation
    functions (resolve_numba_threads, validate_shading_coupling_compat) can issue further
    parser.error() calls without requiring callers to hold a reference to the parser.

    @param[in] args  argument list to parse; defaults to sys.argv[1:] when None
    @returns validated Namespace
    """
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--version",
        action="store_true",
        help="Show the EBFM version and exit.",
    )

    # Parse --version early so users can request it without providing required runtime arguments.
    pre_args, _ = pre_parser.parse_known_args(args)
    if pre_args.version:
        ebfm.core.print_version_and_exit()

    global _parser
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _parser = parser
    mesh_opts = {
        grid_type: f"--{arg_dest.replace('_', '-')}" for grid_type, arg_dest in GridConfig.mesh_arg_dests.items()
    }
    partitioned_elmer_mesh_opt = "--is-partitioned-elmer-mesh"
    netcdf_mesh_opt = "--netcdf-mesh"

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the EBFM version and exit.",
    )

    input_group = parser.add_argument_group("input mesh types")

    primary_grid_group = input_group.add_mutually_exclusive_group(required=True)

    primary_grid_group.add_argument(
        mesh_opts[GridInputType.MATLAB],
        type=Path,
        help="Path to the MATLAB mesh file.",
    )

    primary_grid_group.add_argument(
        mesh_opts[GridInputType.ELMER],
        type=Path,
        help="Path to the Elmer mesh file.",
    )

    input_group.add_argument(
        netcdf_mesh_opt,
        type=Path,
        help="Path to the NetCDF mesh file. Optional if using --elmer-mesh. "
        "If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    input_group.add_argument(
        "--netcdf-mesh-unstructured",
        type=Path,
        help="Path to the unstructured NetCDF mesh file. "
        f"Optional if using {mesh_opts[GridInputType.ELMER]}. "
        f"If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
    )

    grid_types_without_shading = set(GridConfig.mesh_arg_dests.keys()) - GridConfig.grid_types_supporting_shading
    shading_default_info = "(default: True for {}, False for {})".format(
        ", ".join(mesh_opts[g] for g in GridConfig.grid_types_supporting_shading),
        ", ".join(mesh_opts[g] for g in grid_types_without_shading),
    )

    input_group.add_argument(
        "--shading",
        default=argparse.SUPPRESS,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable shading. " + shading_default_info,
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
        help="Start time of the simulation in ISO8601 format",
        default=CliDefaults.START_TIME.value.isoformat(),
    )

    time_group.add_argument(
        "--end-time",
        type=str,
        help="End time of the simulation in ISO8601 format",
        default=CliDefaults.END_TIME.value.isoformat(),
    )

    def parse_time_step(value: str) -> float | str:
        try:
            return float(value)
        except ValueError:
            return value

    time_group.add_argument(
        "--time-step",
        type=parse_time_step,
        help="Time step of the simulation in ISO8601 format. "
        "Note: The difference between --end-time and --start-time must be divisible by --time-step",
        default=CliDefaults.TIME_STEP_SIZE.value,
    )

    time_group.add_argument(
        "--calendar",
        type=str,
        choices=[cal.value for cal in Calendar],
        help=f"Calendar type for time handling. Supported values: {[cal.value for cal in Calendar]}.",
        default=CliDefaults.CALENDAR.value,
    )

    parallel_group = parser.add_argument_group(
        "parallel runs and distributed meshes (requires MPI; install via: pip install 'ebfm[mpi]')"
    )

    parallel_group.add_argument(
        "--local-group-label",
        type=str,
        default=argparse.SUPPRESS,
        help="MPI group label for the local EBFM communicator. Defaults to --component-name.",
    )

    parallel_group.add_argument(
        partitioned_elmer_mesh_opt,
        action="store_true",
        help="Indicate if the provided Elmer mesh is partitioned for parallel runs.",
    )

    parallel_group.add_argument(
        "--use-part",
        type=int,
        default=argparse.SUPPRESS,  # To not print (default: ...) in --help
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

    diagnostics_group = parser.add_argument_group("diagnostics, reference snapshots, random seed")

    diagnostics_group.add_argument(
        "--print-diagnostics",
        action="store_true",
        default=False,
        help=(
            "Log diagnostic info each timestep via the INFO logger: "
            "gpsum, shading status, and smb / smb_cumulative statistics."
        ),
    )

    diagnostics_group.add_argument(
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

    diagnostics_group.add_argument(
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

    args = parser.parse_args(args)

    # Option compatibility checks — all user-facing, produce actionable error messages.
    if args.is_partitioned_elmer_mesh and getattr(args, GridConfig.mesh_arg_dests[GridInputType.ELMER], None) is None:
        parser.error(f"{partitioned_elmer_mesh_opt} requires {mesh_opts[GridInputType.ELMER]}")

    if args.is_partitioned_elmer_mesh and args.netcdf_mesh is None:
        parser.error(f"{partitioned_elmer_mesh_opt} requires {netcdf_mesh_opt}")

    if args.elmer_mesh and args.elmer_mesh_crs_epsg is None:
        parser.error("--elmer-mesh-crs-epsg is required when using --elmer-mesh")

    if not hasattr(args, "local_group_label"):
        args.local_group_label = args.component_name

    _mpi_opts = {
        "--local-group-label": hasattr(args, "local_group_label") and args.local_group_label != args.component_name,
        "--is-partitioned-elmer-mesh": args.is_partitioned_elmer_mesh,
        "--use-part": hasattr(args, "use_part"),
    }
    active_mpi_opts = [opt for opt, active in _mpi_opts.items() if active]
    if active_mpi_opts and not mpi_available:
        parser.error(f"{', '.join(active_mpi_opts)} require MPI. Install via: pip install 'ebfm[mpi]'")

    return args


def resolve_numba_threads(args: Namespace, comm, logger) -> int:
    """Validate numba availability and thread configuration, then return the thread count to use.

    Must be called only when args.with_numba is True. Emits parser.error() for user-input problems
    (missing package, out-of-range thread count) and logs the resolved setting.

    @param[in] args   parsed CLI arguments
    @param[in] comm   MPI communicator (used to determine suggested thread limit)
    @param[in] logger logger for informational messages
    @returns number of Numba threads to initialise
    """
    import multiprocessing as _mp
    from ebfm.core.compute_backend import is_numba_available

    if not is_numba_available():
        _cli_error("--with-numba: numba is not installed. Run: pip install 'ebfm[performance]'")

    cpu_count = _mp.cpu_count()
    mpi_size = comm.size
    suggested_max_threads = max(1, cpu_count // mpi_size)

    n_threads = args.numba_threads if args.numba_threads is not None else 1

    if args.numba_threads is None:
        logger.info(
            f"[NUMBA] --with-numba enabled. Using default threads per rank: {n_threads}. "
            f"To increase, pass --numba-threads N (suggested max: {suggested_max_threads}; "
            f"cpu_count={cpu_count}, MPI world size={mpi_size})."
        )
    else:
        if n_threads < 1:
            _cli_error("--numba-threads must be >= 1")
        if n_threads > suggested_max_threads:
            _cli_error(
                f"[NUMBA] --numba-threads={n_threads} exceeds suggested max {suggested_max_threads} "
                f"(cpu_count={cpu_count}, MPI world size={mpi_size}). "
                f"--numba-threads must be between 1 and {suggested_max_threads} on this system. "
            )
        logger.info(
            f"[NUMBA] --with-numba enabled. Threads per rank: {n_threads} "
            f"(suggested max {suggested_max_threads}; cpu_count={cpu_count}, MPI world size={mpi_size})."
        )

    return n_threads


def validate_shading_coupling_compat(grid_config, coupling_config) -> None:
    """Check that shading and coupling are not enabled simultaneously.

    @param[in] grid_config     resolved GridConfig
    @param[in] coupling_config resolved CouplingConfig
    """
    if grid_config.use_shading and coupling_config.defines_coupling():
        _cli_error(
            "Shading routine not implemented for coupled runs. "
            "Please deactivate shading via --no-shading or deactivate coupling."
        )
