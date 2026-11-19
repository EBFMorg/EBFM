# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read a structured grid from a MATLAB file."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from numpy import dtype, ndarray

logger = logging.getLogger(__name__)


def read_MATLAB_grid(gridfile: Path):
    """
    Provides grid information by reading from a .mat file or allowing user input.

    Parameters:
        gridfile: Path to the .mat file containing grid data.

    Returns:
        dict: A dictionary named `input_data` containing:
            - 'x': 2D NumPy array of UTM easting coordinates (m).
            - 'y': 2D NumPy array of UTM northing coordinates (m).
            - 'z': 2D NumPy array of elevation (m).
            - 'mask': 2D NumPy array of glacier mask (0 = no glacier, 1 = glacier).
    """

    logger.info("EBFM: Reading grid data from MATLAB file...")
    input_data: dict[str, ndarray[tuple[int, ...], dtype[Any]]] = {}

    try:
        # Load MATLAB file
        mat_file = sio.loadmat(str(gridfile))
        grid_svalbard = mat_file.get("grid_svalbard", None)

        if grid_svalbard is None:
            raise ValueError("`grid_svalbard` not found in the .mat file!")

        # Extract data from the MATLAB structure
        input_data["x"] = grid_svalbard["x"]
        input_data["y"] = grid_svalbard["y"]
        input_data["z"] = grid_svalbard["z"]
        input_data["mask"] = grid_svalbard["mask"]

    except FileNotFoundError:
        logger.info(f"File not found: {gridfile}")
        raise
    except KeyError as e:
        logger.info(f"Missing field in .mat file: {e}")
        raise

    return input_data


def read_MATLAB_fields(gridfile: Path) -> dict[str, np.ndarray]:
    """
    The 2-D fields of a MATLAB grid file, oriented so that y and x increase along the first and second axis.

    The flat indices of the active cells (mask == 1) in these fields are the column ids of the grid.

    Returns:
        dict: 'x', 'y', 'z' and 'mask', 2D NumPy arrays as in read_MATLAB_grid.
    """
    input_data = read_MATLAB_grid(gridfile)
    fields = {name: input_data[name][0][0] for name in ("x", "y", "z", "mask")}

    # Flip grid E-W or N-S when needed
    fy, _ = np.gradient(fields["y"])
    if fy[0, 0] < 0:
        fields = {name: np.flipud(values) for name, values in fields.items()}

    _, fx = np.gradient(fields["x"])
    if fx[0, 0] < 0:
        fields = {name: np.fliplr(values) for name, values in fields.items()}

    return fields
