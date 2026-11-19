# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split an EBFM restart file into one restart file per partition of a partitioned grid.

A run on N MPI ranks reads its restart state from a directory restart_<time>/partitioning.N/ that holds part.<n>.nc
for every partition n, each with the columns of its partition. This script writes such a directory from a restart
file of the whole grid, matching the rows of the file to the columns of the partitions by the `column_id` variable.
The partitioning is that of an Elmer mesh (part.<n>.elements) or of a MATLAB or NetCDF grid (part.<n>.mat or .nc).

python -m ebfm.io.merge is the inverse.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

import ebfm.grid.elmer.partitioning as elmer_partitioning
import ebfm.grid.structured.partitioning as structured_partitioning

# The variable that holds the column ids, as in ebfm.core.restart
COLUMN_ID_VARIABLE = "column_id"


def partition_column_ids(partitioning: Path) -> dict[int, np.ndarray]:
    """The column ids of every partition of partitioning, by partition id."""
    if any(Path(partitioning).glob("part.*.elements")):
        return elmer_partitioning.partition_column_ids(partitioning)
    return structured_partitioning.partition_column_ids(partitioning)


def split_restart_file(restart_file: Path, partitioning: Path, out_dir: Path | None = None) -> Path:
    """
    Write out_dir/part.<n>.nc for every partition n of partitioning, with the rows of restart_file for its cells.

    Every variable of the file has the columns along its first axis.

    @param[in] out_dir by default restart_<time>/partitioning.<N> next to restart_file restart_<time>.nc, where a run
                       on N ranks reads it
    @returns out_dir
    """
    with Dataset(restart_file, "r") as src:
        if COLUMN_ID_VARIABLE not in src.variables:
            raise ValueError(
                f"{restart_file} has no '{COLUMN_ID_VARIABLE}' variable, so its rows cannot be matched to the cells "
                "of the partitions."
            )
        variables = {name: np.asarray(var[:]) for name, var in src.variables.items()}

    file_column_ids = variables[COLUMN_ID_VARIABLE]
    sorter = np.argsort(file_column_ids, kind="stable")

    partitions = partition_column_ids(partitioning)
    n_cells = sum(len(ids) for ids in partitions.values())
    if n_cells != len(file_column_ids):
        raise ValueError(
            f"{restart_file} holds {len(file_column_ids)} columns, but {partitioning} has {n_cells} columns."
        )

    if out_dir is None:
        out_dir = Path(restart_file).with_suffix("") / f"partitioning.{len(partitions)}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for partition_id, partition_ids in partitions.items():
        positions = np.minimum(np.searchsorted(file_column_ids, partition_ids, sorter=sorter), len(sorter) - 1)
        rows = sorter[positions]
        is_missing = file_column_ids[rows] != partition_ids
        if is_missing.any():
            raise ValueError(
                f"{restart_file} holds no row for column {partition_ids[is_missing][0]} of partition {partition_id}."
            )

        with Dataset(out_dir / f"part.{partition_id}.nc", "w", format="NETCDF4") as dst:
            for name, values in variables.items():
                _write_variable(dst, name, values[rows] if values.ndim else values)


def _write_variable(ncfile: Dataset, name: str, values: np.ndarray) -> None:
    """Write an array with one dimension of its own per axis, named as EBFM names those of a restart file."""
    dims = []
    for axis, size in enumerate(values.shape):
        dim_name = f"{name}_dim{axis}"
        ncfile.createDimension(dim_name, size)
        dims.append(dim_name)
    ncfile.createVariable(name, values.dtype, tuple(dims))[:] = values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("restart_file", type=Path, help="restart file of the whole grid, restart_<time>.nc")
    parser.add_argument(
        "partitioning", type=Path, help="partitioning of the grid, e.g. MESH/partitioning.4 or GRID.partitioning.4"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="directory to write part.<n>.nc to (default: restart_<time>/partitioning.<N> next to restart_<time>.nc)",
    )
    args = parser.parse_args(argv)

    try:
        out_dir = split_restart_file(args.restart_file, args.partitioning, args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote the restart files of the partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
