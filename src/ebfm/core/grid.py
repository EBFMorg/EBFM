# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
The horizontal grid EBFM integrates on.

A column is point data: a location carrying attributes and a vertical profile, with no area and no
neighbours. Nothing in the time loop integrates over a column's footprint, which is why a grid can be built
from any set of points, and why the cell centres of an Elmer/Ice mesh can be taken as columns directly.

Horizontal geometry therefore exists only as per-column attributes, listed in PER_COLUMN_FIELDS. The one
exception is the shading look-up table, which needs the terrain surrounding each column, including cells
that are not columns themselves. It is built from the 2-D DEM of the MATLAB grid type during initialization;
the time loop only reads one row of it per column.

Every column has an id, `column_id`, that is unique over the whole domain, also when the columns are
distributed over several MPI ranks. For a grid read from an Elmer mesh it is the Elmer element id, for a
MATLAB or NetCDF grid the flat index of the column in the 2-D fields (`ind`). Sorting the columns by id
restores the column order of the whole domain.
"""

from enum import Enum
from typing import Any

import numpy as np


class GridInputType(Enum):
    # .mat grid file with elevation
    MATLAB = "matlab"

    # NetCDF grid file with elevation
    NETCDF = "netcdf"

    # Elmer/Ice mesh file for xy-coordinates and separate NetCDF elevation file
    CUSTOM = "custom"

    # Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS
    ELMERXIOS = "elmerxios"

    # Elmer/Ice mesh file with elevation in z-coordinate
    ELMER = "elmer"


GridDict = dict[str, Any]  # Alias for grid dictionary type, can be replaced with a more specific type in the future


def number_of_columns(grid: GridDict) -> int:
    """
    Number of snow/firn columns the model integrates.

    Parameters:
        grid (GridDict): containing grid-related parameters

    Returns:
        int: Number of columns (gpsum)
    """
    return len(grid["mask"])


# Fields carrying exactly one entry per snow/firn column. Their common length is `gpsum`.
PER_COLUMN_FIELDS = (
    "column_id",
    "x",
    "y",
    "z",
    "lat",
    "lon",
    "mask",
    "slope_x",
    "slope_y",
    "slope_beta",
    "slope_gamma",
)


def validate_grid(grid: GridDict) -> None:
    """
    Check the invariants every grid must satisfy, whichever branch of `init_grid` built it.

    Parameters:
        grid (GridDict): a fully initialised grid

    Raises:
        AssertionError: if any invariant is violated
    """
    mask = grid["mask"]

    # A two-dimensional mask (such as the MATLAB `mask_2D`) would silently make `len(mask)` the number of rows
    # rather than the number of columns, and would let the glacier-cell count exceed it.
    assert mask.ndim == 1, f"Grid mask must be one-dimensional, got shape {mask.shape}."

    gpsum = number_of_columns(grid)
    assert gpsum > 0, "Grid must have at least one column."

    # Compared by value, not dtype: the mask is uint8 (MATLAB), float64 (ELMER, CUSTOM) or int64 (ELMERXIOS).
    assert np.isin(mask, (0, 1)).all(), f"Grid mask must contain only 0 and 1, got values {np.unique(mask)}."

    number_of_glacier_cells = int(np.sum(mask == 1))
    assert (
        number_of_glacier_cells <= gpsum
    ), f"Number of glacier cells ({number_of_glacier_cells}) exceeds the number of columns ({gpsum})."

    # Every per-column field must agree with the mask; otherwise deriving `gpsum` from the mask would size the
    # state arrays inconsistently with the geometry they describe.
    for field in PER_COLUMN_FIELDS:
        assert field in grid, f"Grid is missing the per-column field '{field}'."
        values = grid[field]
        assert values.ndim == 1, f"Per-column field '{field}' must be one-dimensional, got shape {values.shape}."
        assert len(values) == gpsum, f"Per-column field '{field}' has length {len(values)}, expected gpsum={gpsum}."

    assert np.issubdtype(grid["column_id"].dtype, np.integer), "Column ids must be integers."
    assert len(np.unique(grid["column_id"])) == gpsum, "Column ids must be unique."


def domain_mean(values, comm) -> float:
    """
    Mean of a per-column field over the columns of all ranks of comm.

    Collective over comm. With a single rank it is np.mean(values). With several ranks it is the sum of the local
    sums divided by the number of columns, which adds the values in a different order, so it can differ from the
    serial mean in the last bits.

    Parameters:
        values: this rank's values of the field
        comm: EBFM's MPI communicator

    Returns:
        float: the mean, the same on every rank
    """
    if comm.size == 1:
        return np.mean(values)

    # MPI has no mean reduction; sum and count are reduced together, element-wise
    total, count = comm.allreduce(np.array([np.sum(values), len(values)], dtype=np.float64))
    return total / count


class ShadingMethod(Enum):
    """Available shading algorithms for topographic shading.

    Both use a look-up table of maximum horizon angles per column and azimuth sector.

    - `LUT`: EBFM pre-computes the look-up table from the regular 2-D grid during initialization.
    - `LUT_EXTERNAL`: the look-up table is provided by Elmer/Ice. Not implemented yet.
    """

    LUT = "lut_shading"
    LUT_EXTERNAL = "lut_external_shading"
