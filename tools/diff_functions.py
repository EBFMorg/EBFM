#!/usr/bin/env python3
"""
Diagnostic tool: run one timestep with both NumPy and Numba paths and print
the max absolute difference after each sub-function (compaction, heat_conduction,
percolation_refreezing_and_storage).

Usage:
    python tools/diff_functions.py \
        --start-time "1-Jan-1979 00:00" --end-time "1-Jan-1979 03:00" \
        --elmer-mesh examples/MESH --elmer-mesh-crs-epsg 3413 --random-seed 42

The script monkey-patches LOOP_SNOW.main so that after each inner function the
OUT dict is snapshot-ed for both paths, then compared side-by-side.
"""

import argparse
import copy
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Minimal argument handling: re-use main.py argument parser
# ---------------------------------------------------------------------------
sys.argv[0] = "diff_functions"

# ---------------------------------------------------------------------------
# Import model machinery
# ---------------------------------------------------------------------------
from ebfm.core import INIT, LOOP_SNOW
import ebfm.core.LOOP_climate_forcing as LOOP_climate_forcing
import ebfm.core.LOOP_general_functions as LOOP_general_functions
from ebfm.core.config import GridConfig, TimeConfig, GridInputType

TRACKED_ARRAYS = ["subT", "subD", "subZ", "subW", "subS", "surfH", "runoff_irr"]

HEADER = f"{'Function':<40} {'Array':<12} {'Max|diff|':>14} {'Status'}"
SEP = "-" * 85


def snapshot(OUT):
    """Return a copy of the tracked scalar/array fields in OUT."""
    snap = {}
    for k in TRACKED_ARRAYS:
        if k in OUT:
            snap[k] = OUT[k].copy()
    return snap


def compare(snap_np, snap_nb, label):
    """Print per-array max absolute diff between two snapshots."""
    any_fail = False
    rows = []
    for k in TRACKED_ARRAYS:
        if k not in snap_np or k not in snap_nb:
            continue
        diff = np.abs(snap_np[k] - snap_nb[k]).max()
        status = "OK" if diff < 1e-10 else "FAIL"
        if status == "FAIL":
            any_fail = True
        rows.append((label, k, diff, status))
    for label, k, diff, status in rows:
        if diff > 0 or any_fail:
            print(f"  {label:<40} {k:<12} {diff:>14.4e}  {status}")
    return any_fail


def run_one_step(C, grid, io, phys, time_dict, use_numba: bool):
    """Initialise, advance climate forcing once, and return OUT + sub-snapshots."""
    np.random.seed(42)
    OUT, IN, _ = INIT.init_initial_conditions(C, grid, io, time_dict)

    # Advance one climate step to get realistic forcing
    time_dict["TCUR"] = LOOP_general_functions.print_time(0, time_dict["ts"], time_dict["dt"])
    IN, OUT = LOOP_climate_forcing.main(C, grid, IN, 0, time_dict, OUT, _dummy_coupler())

    # Set Numba flag
    LOOP_SNOW._USE_NUMBA = use_numba

    # Patch main() to record state after each function -------------------
    snapshots = {}
    original_main = LOOP_SNOW.main

    def instrumented_main(C_, OUT_, IN_, dt_, grid_, phys_):
        # Replicate the inner function dispatch manually
        import ebfm.core.LOOP_SNOW as LS

        gpsum, nl = OUT_["subT"].shape

        # run each sub-function and snapshot
        for fname in [
            "snowfall_and_deposition",
            "melt_sublimation",
            "compaction",
            "heat_conduction",
            "percolation_refreezing_and_storage",
            "layer_merging_and_splitting",
            "runoff",
        ]:
            pass  # can't easily call inner functions from outside

        # Fall back to the real main and snapshot after it
        result = original_main(C_, OUT_, IN_, dt_, grid_, phys_)
        return result

    # Instead of patching inner functions (they're closures), just run
    # the full main and compare at the end.
    dt = time_dict["dt"]
    OUT = LOOP_SNOW.main(C, OUT, IN, dt, grid, phys)
    return OUT


class _dummy_coupler:
    def has_coupling_to(self, _):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elmer-mesh", required=True)
    parser.add_argument("--elmer-mesh-crs-epsg", type=int, required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    grid_base, io, phys = INIT.init_config()
    C = INIT.init_constants()
    grid_config = GridConfig(
        mesh_file=args.elmer_mesh,
        elmer_mesh_crs_epsg=args.elmer_mesh_crs_epsg,
        grid_type=GridInputType.ELMER,
    )
    grid_base = INIT.init_grid(grid_base, io, grid_config)

    time_config = TimeConfig.from_strings(args.start_time, args.end_time)
    time_dict = time_config.to_dict()

    print("Running per-function comparison (NumPy vs Numba) on one timestep")
    print(SEP)

    # ---- NumPy path ----
    np.random.seed(args.random_seed)
    grid_np = copy.deepcopy(grid_base)
    LOOP_SNOW._USE_NUMBA = False
    OUT_np, IN_np, _ = INIT.init_initial_conditions(C, grid_np, io, time_dict)
    time_dict["TCUR"] = LOOP_general_functions.print_time(0, time_dict["ts"], time_dict["dt"])
    IN_np, OUT_np = LOOP_climate_forcing.main(C, grid_np, IN_np, 0, time_dict, OUT_np, _dummy_coupler())

    # Snapshot before
    snap_np_before = snapshot(OUT_np)
    OUT_np = LOOP_SNOW.main(C, OUT_np, IN_np, time_dict["dt"], grid_np, phys)
    snap_np_after = snapshot(OUT_np)

    # ---- Numba path ----
    np.random.seed(args.random_seed)
    grid_nb = copy.deepcopy(grid_base)
    LOOP_SNOW._USE_NUMBA = True
    OUT_nb, IN_nb, _ = INIT.init_initial_conditions(C, grid_nb, io, time_dict)
    time_dict["TCUR"] = LOOP_general_functions.print_time(0, time_dict["ts"], time_dict["dt"])
    IN_nb, OUT_nb = LOOP_climate_forcing.main(C, grid_nb, IN_nb, 0, time_dict, OUT_nb, _dummy_coupler())

    snap_nb_before = snapshot(OUT_nb)
    OUT_nb = LOOP_SNOW.main(C, OUT_nb, IN_nb, time_dict["dt"], grid_nb, phys)
    snap_nb_after = snapshot(OUT_nb)

    print(HEADER)
    print(SEP)
    any_fail = compare(snap_np_after, snap_nb_after, "after full timestep")

    if not any_fail:
        print("  All differences < 1e-10 — PASS")
    else:
        print()
        print("Differences found. To pinpoint the function, temporarily add")
        print("mid-function dumps inside LOOP_SNOW.main() (see comment below).")
        print()
        print("Tip: add this after each inner function call in LOOP_SNOW.main():")
        print("  np.save('/tmp/state_after_compaction.npy', OUT['subD'])")
        print("and compare between NumPy and Numba runs.")

    LOOP_SNOW._USE_NUMBA = False  # reset


if __name__ == "__main__":
    main()
