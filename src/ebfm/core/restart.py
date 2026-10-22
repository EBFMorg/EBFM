# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""What a restart file contains, and the checks it has to satisfy.

Single source of truth for the state carried across a restart, shared by the writer
(`FINAL_create_restart_file`) and the loader (`INIT.init_initial_conditions`) so the two cannot
drift apart.

A restart file holds the columns of the whole domain, in the order of their ids, and their ids in the variable
`column_id`. A run on several MPI ranks reads and writes it on rank 0, which matches the rows to the columns by id,
so the file does not depend on how the columns are distributed over the ranks.
"""

import logging
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .comm.gather import ColumnGather

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


def read_restart_file(path: Path, nl: int, gather: ColumnGather, grid) -> dict:
    """
    Read a restart file on the root rank of gather and give every rank the rows of its columns.

    Collective over gather.comm. The rows of a file with a `column_id` variable are matched to the columns by id. A
    file without one holds the columns in the order of a run on a single rank, which, for a grid that EBFM splits
    over the ranks, is the order of the ranks' columns concatenated in rank order. On several ranks, the x and y of
    such a file are therefore checked against the grid.

    Parameters:
        path (Path): Restart file to read
        nl (int): Number of layers per column
        gather (ColumnGather): The columns of all ranks
        grid (dict): This rank's part of the grid

    Returns:
        dict: every variable of the file except `column_id`, with the rows of this rank's columns

    Raises:
        ValueError: on the root rank, if the file does not match the columns of all ranks
    """
    is_parallel_run = gather.comm.size > 1

    variables, is_ordered_by_id = None, None
    if gather.is_root:
        n_columns = len(gather.column_ids)
        variables = _read_variables(path, n_columns, nl)
        file_column_ids = variables.pop(COLUMN_ID_VARIABLE, None)
        is_ordered_by_id = file_column_ids is not None
        if is_ordered_by_id:
            rows = _rows_by_column_id(file_column_ids, gather.column_ids, path)
            variables = {name: values[rows] if np.ndim(values) else values for name, values in variables.items()}
        elif is_parallel_run:
            logger.warning(
                f"Restart file {path} has no '{COLUMN_ID_VARIABLE}' variable; assuming its rows are in the order of a "
                "run on a single rank."
            )

    layout = None
    if gather.is_root:
        layout = ([(name, np.ndim(values)) for name, values in variables.items()], is_ordered_by_id)
    if is_parallel_run:
        layout = gather.comm.bcast(layout, root=gather.root)
    names_and_ndims, is_ordered_by_id = layout

    local = {}
    for name, ndim in names_and_ndims:
        values = variables[name] if gather.is_root else None
        if ndim == 0:
            local[name] = gather.comm.bcast(values, root=gather.root) if is_parallel_run else values
        elif is_ordered_by_id:
            local[name] = gather.scatter(values)
        else:
            local[name] = gather.scatter_unordered(values)

    if is_parallel_run and not is_ordered_by_id:
        _check_coordinates(local, grid, path)
    return local


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


def _check_coordinates(variables: dict, grid, path: Path) -> None:
    """Check that the x and y of the restart file are those of this rank's columns, to 1 mm."""
    for coordinate in ("x", "y"):
        if not np.allclose(variables[coordinate], grid[coordinate], rtol=0, atol=1e-3):
            raise ValueError(
                f"The '{coordinate}' of restart file {path} differs from that of the grid. The file has no "
                f"'{COLUMN_ID_VARIABLE}' variable, so its rows are assumed to be in the order of a run on a single "
                "rank; it was written for a different grid or a different order of the columns."
            )
