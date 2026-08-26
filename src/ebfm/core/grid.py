# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from typing import Any

import numpy as np


class GridInputType(Enum):
    # .mat grid file with elevation
    MATLAB = "matlab"

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
PER_COLUMN_FIELDS = ("x", "y", "z", "lat", "lon", "mask", "slope_x", "slope_y", "slope_beta", "slope_gamma")


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


class ShadingMethod(Enum):
    """Available shading algorithms for topographic shading.

    - `CLASSICAL`: computes shading online each time step by ray-marching from
        each glacier cell in the current solar azimuth direction.
    - `LUT`: uses a precomputed look-up table of maximum horizon angles per cell
        and azimuth sector, which is faster at runtime.
    """

    CLASSICAL = "classical_shading"
    LUT = "lut_shading"
