# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split an EBFM restart file into one restart file per partition of a partitioned Elmer mesh.

A run on a partitioned Elmer mesh (--is-partitioned-elmer-mesh) reads its restart state from a directory
restart_<time>/ that holds part.<n>.nc for every partition n, each with the columns of its partition. This script
writes such a directory from a restart file of the whole mesh, matching the rows of the file to the cells of the
partitions by the `column_id` variable, the Elmer element id.

python -m ebfm.io.merge is the inverse.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# The variable that holds the column ids, as in ebfm.core.restart
COLUMN_ID_VARIABLE = "column_id"


def partition_element_ids(mesh_dir: Path) -> dict[int, np.ndarray]:
    """The element ids of every partition of a partitioned Elmer mesh, by partition id, in file order."""
    element_files = {int(f.name.split(".")[1]): f for f in mesh_dir.glob("part.*.elements")}
    if not element_files:
        raise ValueError(f"{mesh_dir} holds no part.<n>.elements files of a partitioned Elmer mesh.")
    if sorted(element_files) != list(range(1, len(element_files) + 1)):
        raise ValueError(f"The partitions of {mesh_dir} are not numbered 1 to {len(element_files)}.")
    return {
        partition_id: np.loadtxt(element_files[partition_id], usecols=0, dtype=np.int64, ndmin=1)
        for partition_id in sorted(element_files)
    }


def split_restart_file(restart_file: Path, mesh_dir: Path, out_dir: Path) -> None:
    """
    Write out_dir/part.<n>.nc for every partition n of mesh_dir, with the rows of restart_file for its cells.

    Every variable of the file has the columns along its first axis.
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

    partitions = partition_element_ids(mesh_dir)
    n_cells = sum(len(ids) for ids in partitions.values())
    if n_cells != len(file_column_ids):
        raise ValueError(f"{restart_file} holds {len(file_column_ids)} columns, but {mesh_dir} has {n_cells} cells.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for partition_id, element_ids in partitions.items():
        positions = np.minimum(np.searchsorted(file_column_ids, element_ids, sorter=sorter), len(sorter) - 1)
        rows = sorter[positions]
        is_missing = file_column_ids[rows] != element_ids
        if is_missing.any():
            raise ValueError(
                f"{restart_file} holds no row for element {element_ids[is_missing][0]} of partition {partition_id}."
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
    parser.add_argument("restart_file", type=Path, help="restart file of the whole mesh, restart_<time>.nc")
    parser.add_argument("mesh_dir", type=Path, help="directory of the partitioned Elmer mesh (part.<n>.elements)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="directory to write part.<n>.nc to (default: the restart file's path without .nc, restart_<time>/)",
    )
    args = parser.parse_args(argv)

    out_dir = args.output or args.restart_file.with_suffix("")
    try:
        split_restart_file(args.restart_file, args.mesh_dir, out_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote the restart files of the partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
