# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import argparse

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

from typing import List

# logger for this module
logger: Logger = None  # will be set later


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
        default=FieldValidationLevel.FATAL.value,
        help="Level of validation for field exchange type checks. "
        "'FATAL': raise exception on mismatch (default), "
        "'WARNING': log warning on mismatch, "
        "'SILENT': only log at debug level on mismatch.",
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


# Arrays saved by --dump-reference
_REFERENCE_KEYS = [
    "smb", "smb_cumulative", "Tsurf",
    "subT", "subD", "subZ", "subW", "subS", "surfH",
    "subTmean", "runoff_irr",
    "Dens_destr_metam", "Dens_overb_pres", "Dens_drift",
    # reboot / restart state
    "snowmass", "ys", "timelastsnow_netCDF", "alb_snow",
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
        logger.info(
            f"[DIAG t={t + 1}] smb:            min={smb.min():.4e}  max={smb.max():.4e}  mean={smb.mean():.4e}"
        )
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
        "--netcdf-mesh-unstructured",
        type=Path,
        help="Path to the unstructured NetCDF mesh file. Optional if using --elmer-mesh."
        " If --netcdf-mesh is provided elevations will be read from the given NetCDF mesh file.",
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
        help="Start time of the simulation in format 'DD-Mon-YYYY HH:MM' "
        "(i.e., time at the beginning of the first time step)",
        default="1-Jan-1979 00:00",
    )

    time_group.add_argument(
        "--end-time",
        type=str,
        help="End time of the simulation in format 'DD-Mon-YYYY HH:MM' "
        "(i.e., time at the end of the last time step)",
        default="2-Jan-1979 00:00",
    )

    time_group.add_argument(
        "--time-step",
        type=float,
        help="Time step of the simulation in days, e.g., 0.125 for 3 hours. "
        "Note: The difference between end-time and start-time must be divisible by the time step.",
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

    logger_group = parser.add_argument_group("logging configuration")

    logger_group.add_argument(
        "--log-level-console",
        type=str,
        choices=list(log_levels_map.keys()),
        default="INFO",
        help="Log level for console output for all MPI ranks (unless overridden by custom settings in utils.py).",
    )

    logger_group.add_argument(
        "--log-file",
        type=Path,
        help="If provided, log output will be written to the specified file (one file per MPI rank).",
    )

    diag_group = parser.add_argument_group("diagnostics and profiling")

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
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Wrap the run with cProfile and log the top-30 hotspots by cumulative time at the end. "
            "For line-level profiling of heat_conduction(), run with kernprof instead: "
            "kernprof -l src/main.py <args> && python -m line_profiler main.py.lprof"
        ),
    )

    diag_group.add_argument(
        "--dump-reference",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "After the final timestep, save key output arrays (smb, smb_cumulative, "
            "Tsurf, subT, subD, subZ) to FILE as a NumPy .npz archive. "
            "Use tools/compare_snapshots.py to diff two such files."
        ),
    )

    diag_group.add_argument(
        "--no-shading",
        action="store_true",
        default=False,
        help=(
            "Disable the topographic shading routine even when using a MATLAB mesh. "
            "Useful for performance comparisons: shading adds up to 200 ray-tracing "
            "iterations per timestep but does not couple grid points to each other."
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

    diag_group.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of Numba threads to use per MPI rank for the parallel heat_conduction() "
            "kernel (requires numba; install via 'pip install ebfm[performance]'). "
            "Defaults to cpu_count // mpi_world_size to avoid CPU oversubscription when "
            "running multiple MPI ranks on a shared node. Has no effect without numba."
        ),
    )

    diag_group.add_argument(
        "--with-numba",
        action="store_true",
        default=False,
        help=(
            "Enable the parallel Numba kernel for heat_conduction() (opt-in). "
            "Requires numba; install via: pip install 'ebfm[performance]'. "
            "Without this flag the fast NumPy A+B+C path is used regardless of whether "
            "numba is installed. Do NOT use NUMBA_DISABLE_JIT=1 to benchmark the NumPy "
            "path: that env-var makes @njit a no-op so the kernel runs as plain Python "
            "loops which are ~100x slower than NumPy."
        ),
    )

    diag_group.add_argument(
        "--diag-dump",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Save per-function diagnostic snapshots (subT, subD, subZ, subW, subS, surfH) "
            "to DIR after every inner function call in LOOP_SNOW.main(). "
            "Files are named step<NNN>_<function>.npz. "
            "Run twice (once NumPy, once with --with-numba) to two different directories, "
            "then compare matching files with tools/compare_snapshots.py or numpy.testing."
        ),
    )

    # Add args for features requiring 'import coupling'
    add_coupling_arguments(parser)

    args = parser.parse_args()

    if args.version:
        ebfm.core.print_version_and_exit()

    # Validate that --elmer-mesh-crs-epsg is provided when using --elmer-mesh
    if args.elmer_mesh and args.elmer_mesh_crs_epsg is None:
        parser.error("--elmer-mesh-crs-epsg is required when using --elmer-mesh")

    has_active_coupling_features = extract_active_coupling_features(args)
    if has_active_coupling_features and not ebfm.coupling.coupling_supported:
        raise RuntimeError(
            f"""
Coupling requested via command line argument(s) {has_active_coupling_features}, but the 'coupling' module could not be
imported due to the following error:

{ebfm.coupling.coupling_supported_import_error}

Hint: If you are missing 'yac', please install YAC and the python bindings as described under
https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html"
"""
        )

    # TODO: replace MPI.COMM_WORLD with communicator from ebfm; either from couplers comm splitting or default comm
    setup_logging(
        stdout_log_level=log_levels_map[args.log_level_console],
        file=args.log_file,
        comm=MPI.COMM_WORLD,
    )

    logger = getLogger(__name__)
    logger.info(f"Starting EBFM version {ebfm.core.get_version()}...")

    # Numba is opt-in: only activate when --with-numba is explicitly passed.
    # set_num_threads() must be called before any prange() kernel runs.
    if args.with_numba:
        if not LOOP_SNOW._NUMBA_AVAILABLE:
            parser.error("--with-numba: numba is not installed. " "Run: pip install 'ebfm[performance]'")
        import multiprocessing as _mp
        import numba as _numba_mod

        _n_threads = (
            args.numba_threads if args.numba_threads is not None else max(1, _mp.cpu_count() // MPI.COMM_WORLD.size)
        )
        _numba_mod.set_num_threads(_n_threads)
        LOOP_SNOW._USE_NUMBA = True
        logger.info(
            f"[NUMBA] --with-numba: parallel Numba kernels enabled — "
            f"threads per rank: {_n_threads} "
            f"(cpu_count={_mp.cpu_count()}, MPI world size={MPI.COMM_WORLD.size}). "
            f"First-call JIT cost is amortised by cache=True."
        )
    else:
        logger.info(
            "Pass --with-numba to enable the parallel Numba kernels " "(requires: pip install 'ebfm[performance]')."
        )

    if args.diag_dump is not None:
        import os as _os

        _os.makedirs(args.diag_dump, exist_ok=True)
        LOOP_SNOW._DIAG_DUMP = args.diag_dump
        LOOP_SNOW._DIAG_STEP = 0
        logger.info(f"[DIAG] Per-function dumps enabled → {args.diag_dump}")

    if args.profile:
        import cProfile as _cProfile
        import io as _io
        import pstats as _pstats

        _pr = _cProfile.Profile()
        _pr.enable()
        logger.info("[PROFILE] cProfile enabled.")

    logger.info("Done parsing command line arguments.")
    logger.debug("Parsed the following command line arguments:")
    for arg, val in vars(args).items():
        logger.debug(f"  {arg}: {val}")

    logger.debug("Reading configuration and checking for consistency.")

    # TODO consider introducing an ebfm_adapter_config.yaml to be parsed alternatively/additionally to command line args
    coupling_config = CouplingConfig(args, component_name="ebfm")  # TODO: get from EBFM's coupling configuration?
    grid_config = GridConfig(args)
    time_config = TimeConfig(args)

    logger.debug("Successfully completed consistency checks.")

    # Model setup & initialization
    grid, io, phys = INIT.init_config()
    time = time_config.to_dict()

    C = INIT.init_constants()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        logger.info(f"[DIAG] NumPy random seed fixed to {args.random_seed} for reproducibility.")

    grid = INIT.init_grid(grid, io, grid_config)

    if args.no_shading and grid["has_shading"]:
        logger.info("[DIAG] --no-shading flag set: overriding has_shading=True -> False.")
        grid["has_shading"] = False

    # Ensure shading routine is only used in uncoupled runs on unpartitioned MATLAB grids;
    # see https://github.com/EBFMorg/EBFM/issues/11 for details.
    if grid["has_shading"]:
        assert grid_config.is_partitioned is False, "Shading routine only implemented for unpartitioned grids."
        assert grid_config.grid_type is GridInputType.MATLAB, "Shading routine only implemented for MATLAB input grids."
        assert coupling_config.defines_coupling() is False, "Shading routine not implemented for coupled runs."

    OUT, IN, OUTFILE = INIT.init_initial_conditions(C, grid, io, time)

    # TODO: some grids currently do not have grid["mesh"]
    try:
        grid["mesh"]
    except KeyError:
        grid["mesh"] = None  # add dummy to make coupler.setup pass.

    if coupling_config.defines_coupling():
        coupler = ebfm.coupling.YACCoupler(coupling_config=coupling_config)
    else:
        coupler = ebfm.coupling.DummyCoupler(coupling_config=coupling_config)

    coupler.setup(grid["mesh"], time)

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

            data_from_icon, errors_from_icon = icon_atmo.exchange(data_to_icon)

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
            IN["q"] = data_from_icon["huss"] if not errors_from_icon["huss"] else _T0
            IN["Pres"] = data_from_icon["sfcpres"] if not errors_from_icon["sfcpres"] else _T0 + 101500.0

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
            if coupler.has_coupling_to("icon_atmo"):
                grid["z"] = IN["h"][0].ravel()
            # TODO add gradient field later
            # IN['dhdx'] = data_from_elmer('dhdx')
            # IN['dhdy'] = data_from_elmer('dhdy')

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
    if not grid["is_partitioned"] and isinstance(coupler, ebfm.coupling.DummyCoupler):
        FINAL_create_restart_file.main(OUT, io)
    else:
        logger.warning("Skipping writing of restart file for coupled and/or partitioned runs.")

    logger.info("Time loop completed.")

    if args.dump_reference:
        dump_reference(logger, OUT, args.dump_reference)

    if args.profile:
        _pr.disable()
        _s = _io.StringIO()
        _ps = _pstats.Stats(_pr, stream=_s).sort_stats("cumulative")
        _ps.print_stats(30)
        logger.info("[PROFILE] cProfile top 30 by cumulative time:\n" + _s.getvalue())

    coupler.finalize()

    logger.info("Closing down EBFM.")


# Entry point for script execution
if __name__ == "__main__":
    main()
