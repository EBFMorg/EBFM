# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge the per-partition files of an EBFM run on a partitioned grid into one file.

A run on several MPI ranks writes a directory instead of a file, with part.<n>.nc for every partition n:

* output_<time>/: output files. For an Elmer mesh, the merged file holds the cells of all partitions, ordered by
  column id, and every vertex once, ordered by vertex id. For a MATLAB or NetCDF grid, it holds the (y, x) fields
  with the values of all partitions.
* restart_<time>/: restart files. The merged file is a restart file of the whole grid, ordered by column id, which
  every run can start from.

python -m ebfm.io.split_restart is the inverse for restart files.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# The variable that holds the column ids, as in ebfm.core.restart
COLUMN_ID_VARIABLE = "column_id"

VERTEX_ID_VARIABLE = "vertex_id"


def partition_files(directory: Path) -> list[Path]:
    """The part.<n>.nc files of directory, ordered by n."""
    files = {}
    for f in directory.glob("part.*.nc"):
        match = re.fullmatch(r"part\.(\d+)\.nc", f.name)
        if match:
            files[int(match.group(1))] = f
    if not files:
        raise ValueError(f"{directory} holds no part.<n>.nc files.")
    return [files[n] for n in sorted(files)]


def _read(path: Path) -> tuple[dict, dict, dict, dict]:
    """The dimensions, variables, their attributes and the global attributes of a file."""
    with Dataset(path, "r") as nc:
        dims = {name: (None if dim.isunlimited() else len(dim)) for name, dim in nc.dimensions.items()}
        variables = {name: (var.dimensions, np.asarray(var[:])) for name, var in nc.variables.items()}
        attributes = {name: {a: var.getncattr(a) for a in var.ncattrs()} for name, var in nc.variables.items()}
        global_attributes = {a: nc.getncattr(a) for a in nc.ncattrs()}
    return dims, variables, attributes, global_attributes


def _check_ids(ids: np.ndarray, what: str, directory: Path) -> None:
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Several files in {directory} hold the same {what}; they belong to different runs.")


def merge_restart_files(files: list[Path], directory: Path) -> tuple[dict, dict, dict, dict]:
    """
    Concatenate the restart files of the partitions along the columns, ordered by column id.

    Every variable of a restart file has the columns along its first axis and one dimension of its own per axis.
    """
    parts = [_read(f)[1] for f in files]
    for f, part in zip(files, parts):
        if COLUMN_ID_VARIABLE not in part:
            raise ValueError(f"{f} has no '{COLUMN_ID_VARIABLE}' variable.")

    column_ids = np.concatenate([part[COLUMN_ID_VARIABLE][1] for part in parts])
    _check_ids(column_ids, "column", directory)
    order = np.argsort(column_ids, kind="stable")

    variables, dims = {}, {}
    for name, (var_dims, _) in parts[0].items():
        values = np.concatenate([part[name][1] for part in parts])[order]
        variables[name] = (var_dims, values)
        dims.update(zip(var_dims, values.shape))
    return dims, variables, {}, {}


def merge_output_files(files: list[Path], directory: Path) -> tuple[dict, dict, dict, dict]:
    """
    Concatenate the output files of the partitions along their cells, ordered by column id, and merge their
    vertices, ordered by vertex id. The time axis is that of the first file.
    """
    parts = [_read(f) for f in files]
    first_dims, first_variables, attributes, global_attributes = parts[0]
    part_variables = [variables for _, variables, _, _ in parts]
    for f, variables in zip(files, part_variables):
        if COLUMN_ID_VARIABLE not in variables:
            raise ValueError(f"{f} has no '{COLUMN_ID_VARIABLE}' variable; it was not written on a partitioned mesh.")

    def concatenate(name: str, axis: int = 0) -> np.ndarray:
        return np.concatenate([variables[name][1] for variables in part_variables], axis=axis)

    column_ids = concatenate(COLUMN_ID_VARIABLE)
    _check_ids(column_ids, "column", directory)
    cell_order = np.argsort(column_ids, kind="stable")

    dims = dict(first_dims)
    dims["cell"] = len(column_ids)
    if VERTEX_ID_VARIABLE in first_variables:
        vertex_ids, first_occurrence = np.unique(concatenate(VERTEX_ID_VARIABLE), return_index=True)
        dims["vertex"] = len(vertex_ids)

    merged = {}
    for name, (var_dims, values) in first_variables.items():
        if name == "cell_vertices":
            # The indices into the vertices of each file, as vertex ids, as indices into the merged vertices
            cell_vertex_ids = np.concatenate(
                [variables[VERTEX_ID_VARIABLE][1][variables[name][1]] for variables in part_variables]
            )
            values = np.searchsorted(vertex_ids, cell_vertex_ids[cell_order]).astype(np.int32)
        elif "cell" in var_dims:
            axis = var_dims.index("cell")
            values = np.take(concatenate(name, axis), cell_order, axis=axis)
        elif "vertex" in var_dims:
            values = concatenate(name)[first_occurrence]
        merged[name] = (var_dims, values)
    return dims, merged, attributes, global_attributes


def merge_structured_output_files(files: list[Path], directory: Path) -> tuple[dict, dict, dict, dict]:
    """
    Combine the output files of the partitions of a structured grid, whose (y, x) fields hold values only at the
    cells of their own columns: each partition contributes the values at its columns, the grid fields and the time
    axis are those of the first file.
    """
    parts = [_read(f) for f in files]
    first_dims, first_variables, attributes, global_attributes = parts[0]
    part_variables = [variables for _, variables, _, _ in parts]
    for f, variables in zip(files, part_variables):
        if COLUMN_ID_VARIABLE not in variables:
            raise ValueError(f"{f} has no '{COLUMN_ID_VARIABLE}' variable; it was not written on a partitioned grid.")

    column_ids = np.concatenate([variables[COLUMN_ID_VARIABLE][1] for variables in part_variables])
    _check_ids(column_ids, "column", directory)

    dims = dict(first_dims)
    dims["column"] = len(column_ids)
    merged = {}
    for name, (var_dims, values) in first_variables.items():
        if name == COLUMN_ID_VARIABLE:
            values = np.sort(column_ids)
        elif "time" in var_dims and "y" in var_dims and var_dims[var_dims.index("y") + 1 :][:1] == ("x",):
            # Index the cells of the (y, x) plane by their flat index, the column id
            axis = var_dims.index("y")
            values = values.copy()
            flat = values.reshape(values.shape[:axis] + (-1,) + values.shape[axis + 2 :])
            for variables in part_variables:
                part_values = variables[name][1]
                part_flat = part_values.reshape(part_values.shape[:axis] + (-1,) + part_values.shape[axis + 2 :])
                at_columns = (slice(None),) * axis + (variables[COLUMN_ID_VARIABLE][1],)
                flat[at_columns] = part_flat[at_columns]
        merged[name] = (var_dims, values)
    return dims, merged, attributes, global_attributes


def merge_partition_files(directory: Path, out_file: Path) -> None:
    """Merge the part.<n>.nc files of directory, output or restart files, into out_file."""
    files = partition_files(directory)
    with Dataset(files[0], "r") as nc:
        dimensions = set(nc.dimensions)
    if "cell" in dimensions:
        merge = merge_output_files
    elif {"time", "y", "x"} <= dimensions:
        merge = merge_structured_output_files
    else:
        merge = merge_restart_files
    dims, variables, attributes, global_attributes = merge(files, directory)

    with Dataset(out_file, "w", format="NETCDF4") as dst:
        dst.setncatts(global_attributes)
        for name, size in dims.items():
            dst.createDimension(name, size)
        for name, (var_dims, values) in variables.items():
            var_attributes = dict(attributes.get(name, {}))
            fill_value = var_attributes.pop("_FillValue", None)
            var = dst.createVariable(name, values.dtype, var_dims, fill_value=fill_value)
            var.setncatts(var_attributes)
            var[:] = values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directory", type=Path, help="directory of part.<n>.nc files, output_<time>/ or restart_<time>/"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="file to write (default: the directory's path with .nc, output_<time>.nc or restart_<time>.nc)",
    )
    args = parser.parse_args(argv)

    directory = args.directory.resolve()
    out_file = args.output or directory.with_name(directory.name + ".nc")
    try:
        merge_partition_files(directory, out_file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote the merged file {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
