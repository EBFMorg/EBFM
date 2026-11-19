# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""What a restart file contains, and the checks it has to satisfy.

Single source of truth for the state carried across a restart, shared by the writer
(`FINAL_create_restart_file`) and the loader (`INIT.init_initial_conditions`) so the two cannot
drift apart.

A restart file holds the columns of the whole domain, in the order of their ids, and their ids in the variable
`column_id`, by which a run matches its rows to its columns.

A run on N MPI ranks writes and reads a directory of restart files instead, `restart_<time>/partitioning.N/part.<n>.nc`
for partition n, each with the columns of its partition. The modules
ebfm.io.split_restart and ebfm.io.merge convert between the two.
"""

import logging
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

logger = logging.getLogger(__name__)

# Ids of the columns the rows of a restart file belong to (grid["column_id"]); older files lack it
COLUMN_ID_VARIABLE = "column_id"

# One value per snow/firn column: shape `(gpsum,)`.
PER_COLUMN_VARIABLES = (
    "snowmass",
    "Tsurf",
    "ys",
    "timelastsnow_netCDF",
    "alb_snow",
    "surface_elevation",
    "x",
    "y",
)

# One value per layer of every column: shape `(gpsum, nl)`.
PER_LAYER_VARIABLES = ("subZ", "subW", "subD", "subS", "subT", "subTmean")

# Every variable a restart file carries. The order is the order they are written in.
RESTART_VARIABLES = PER_LAYER_VARIABLES + PER_COLUMN_VARIABLES


def partitioning_name(n_partitions: int) -> str:
    """Name of the directory of the files of the partitions of a partitioning into n_partitions."""
    return f"partitioning.{n_partitions}"


def partition_file_name(partition_id: int) -> str:
    """Name of the file of one partition in a directory of per-partition output or restart files."""
    return f"part.{partition_id}.nc"


def expected_shape(var_name: str, gpsum: int, nl: int) -> tuple[int, ...] | None:
    """
    The shape a named restart variable must have.

    Parameters:
        var_name (str): Name of the variable as stored in the restart file
        gpsum (int): Number of columns
        nl (int): Number of layers per column

    Returns:
        tuple | None: the required shape, or None for a variable outside the manifest
    """
    if var_name in PER_LAYER_VARIABLES:
        return (gpsum, nl)
    if var_name in PER_COLUMN_VARIABLES:
        return (gpsum,)
    return None


def validate_variable_shape(var_name: str, shape: tuple[int, ...], gpsum: int, nl: int, source: Path) -> None:
    """
    Check one restart variable against the configured column discretization.

    Variables in the manifest must have exactly the shape their group requires, which is what rules
    out a per-layer array stored per-column or vice versa. Anything else only has to carry a
    recognisable column axis, so a restart file may gain auxiliary variables without changes here.

    Parameters:
        var_name (str): Name of the variable as stored in the restart file
        shape (tuple): Shape the variable actually has in the file
        gpsum (int): Number of columns
        nl (int): Number of layers per column
        source (Path): Restart file the variable came from, for the error message

    Raises:
        ValueError: if the shape does not match
    """
    required = expected_shape(var_name, gpsum, nl)
    if required is not None:
        if shape != required:
            raise ValueError(
                f"Restart variable '{var_name}' in {source} has shape {shape}, " f"but must have shape {required}."
            )
        return

    if len(shape) > 2:
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has {len(shape)} dimensions; "
            "restart files hold per-column and per-layer variables only."
        )
    if len(shape) == 1 and shape != (gpsum,):
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has shape {shape}, "
            f"but every per-column variable must have shape {(gpsum,)}."
        )
    if len(shape) == 2 and shape != (gpsum, nl):
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has shape {shape}, "
            f"but every per-layer variable must have shape {(gpsum, nl)}."
        )


def validate_all_variables_present(var_names, source: Path) -> None:
    """
    Check that a restart file carries every variable the model needs to resume.

    A missing variable is otherwise only noticed when the time loop indexes it, far from the cause.

    Parameters:
        var_names (Iterable[str]): Names present in the restart file
        source (Path): Restart file being loaded, for the error message

    Raises:
        ValueError: if any manifest variable is absent
    """
    missing = [name for name in RESTART_VARIABLES if name not in set(var_names)]
    if missing:
        raise ValueError(f"Restart file {source} is missing the required variable(s): {', '.join(missing)}.")


def read_restart_file(path: Path, nl: int, grid, partition_id: int | None = None) -> dict:
    """
    Read the rows of this rank's columns from a restart file.

    A run on a single rank reads a restart file of the whole grid. Its rows are matched to the columns by the
    `column_id` variable; a file without it, written by an older version of EBFM, holds them in the order of the grid.
    A run on several ranks reads a directory restart_<time>/partitioning.N instead, of which every rank reads the file
    of its own partition.

    Parameters:
        path (Path): Restart file, or directory of the restart files of a partitioning, to read
        nl (int): Number of layers per column
        grid (dict): This rank's part of the grid
        partition_id (int | None): Partition of a partitioned grid that this rank computes; None for other grids

    Returns:
        dict: every variable of the file except `column_id`, with the rows of this rank's columns

    Raises:
        ValueError: if the file does not match the columns
    """
    if path.is_dir():
        if partition_id is None:
            raise ValueError(
                f"{path} is a directory of per-partition restart files, which only a run on several MPI ranks reads. "
                "Merge them into one restart file with python -m ebfm.io.merge."
            )
        return _read_partition_file(path / partition_file_name(partition_id), nl, grid["column_id"])

    column_ids = grid["column_id"]
    variables = _read_variables(path, len(column_ids), nl)
    file_column_ids = variables.pop(COLUMN_ID_VARIABLE, None)
    if file_column_ids is None:
        return variables
    rows = _rows_by_column_id(file_column_ids, column_ids, path)
    return {name: values[rows] if np.ndim(values) else values for name, values in variables.items()}


def _read_partition_file(path: Path, nl: int, column_ids) -> dict:
    """Read the restart file of one partition, with the rows in the order of column_ids."""
    if not path.exists():
        raise ValueError(
            f"There is no restart file {path} for this partition, so the files of the partitioning in {path.parent} "
            "are incomplete."
        )

    with Dataset(path, "r") as ncfile:
        if COLUMN_ID_VARIABLE not in ncfile.variables:
            raise ValueError(f"Restart file {path} of a partition has no '{COLUMN_ID_VARIABLE}' variable.")
        n_file_columns = len(ncfile.variables[COLUMN_ID_VARIABLE])
    if n_file_columns != len(column_ids):
        raise ValueError(
            f"Restart file {path} holds {n_file_columns} columns, but this partition has {len(column_ids)}. It was "
            "written for a different partitioning; merge the directory into one restart file with 'python -m "
            "ebfm.io.merge', which any run can start from."
        )

    variables = _read_variables(path, len(column_ids), nl)
    rows = _rows_by_column_id(variables.pop(COLUMN_ID_VARIABLE), column_ids, path)
    return {name: values[rows] if np.ndim(values) else values for name, values in variables.items()}


def _read_variables(path: Path, n_columns: int, nl: int) -> dict:
    """Read and check every variable of a restart file for n_columns columns of nl layers."""
    variables = {}
    with Dataset(path, "r") as ncfile:
        validate_all_variables_present(ncfile.variables, path)
        for var_name in ncfile.variables:
            # netCDF4 returns a numpy.ma.MaskedArray whenever the variable defines a fill value, regardless of whether
            # any missing values are actually present.
            var_data = ncfile.variables[var_name][:]
            # Restart files are expected to be fully populated. Mixing masked arrays into the simulation state is
            # unsafe: numpy's masked-array ufuncs silently ignore `where=` when combining masks (only the data
            # respects `where`), so masks can leak into cells they should not and snowball across timesteps. Verify
            # the assumption holds, then drop down to a plain ndarray.
            assert not np.ma.is_masked(
                var_data
            ), f"Restart variable '{var_name}' in {path} contains missing values, which is not supported."
            if isinstance(var_data, np.ma.MaskedArray):
                var_data = var_data.data

            validate_variable_shape(var_name, var_data.shape, n_columns, nl, path)

            # A variable without dimensions becomes a Python scalar
            variables[var_name] = var_data.item() if var_data.shape == () else var_data
    return variables


def _rows_by_column_id(file_column_ids, column_ids, path: Path):
    """The row of the file for each of column_ids. Both hold the same number of ids."""
    sorter = np.argsort(file_column_ids, kind="stable")
    positions = np.minimum(np.searchsorted(file_column_ids, column_ids, sorter=sorter), len(sorter) - 1)
    rows = sorter[positions]

    is_missing = file_column_ids[rows] != column_ids
    if is_missing.any():
        raise ValueError(
            f"Restart file {path} holds no row for {is_missing.sum()} of the {len(column_ids)} columns, e.g. for "
            f"column id {column_ids[is_missing][0]}. It was written for a different grid."
        )
    return rows
