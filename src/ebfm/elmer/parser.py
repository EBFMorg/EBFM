# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray

import pandas as pd
from pathlib import Path

"""This file contains parsers for different file formats used in Elmer
"""


def parse_header(header_file: Path) -> tuple[int, int]:
    """Parse Elmer mesh header file.
    Args:
        header_file (Path): Path to the Elmer mesh header file.
    Returns:
        tuple: Number of vertices and cells.
    """
    df: pd.DataFrame = pd.read_csv(header_file, sep="\\s+", header=None, dtype=int, nrows=1)

    # A valid mesh.header file in Elmer should have exactly 3 columns:
    # 1. Number of vertices
    # 2. Number of 303 cells
    # 3. Number of 202 boundary edges (will be ignored)
    n_expected_columns: int = 3
    if df.shape[1] != n_expected_columns:
        raise ValueError(
            f"The header file has only {df.shape[1]} columns, but it should have exactly {n_expected_columns}."
        )

    if isinstance(df.iloc[0, 0], np.int_):
        n_vertices: int = int(df.iloc[0, 0])
    else:
        raise ValueError(
            f"Expected the first column of the header file to contain an integer, but got {type(df.iloc[0, 0])}."
        )

    if isinstance(df.iloc[0, 1], np.int_):
        n_cells: int = int(df.iloc[0, 1])
    else:
        raise ValueError(
            f"Expected the second column of the header file to contain an integer, but got {type(df.iloc[0, 1])}."
        )

    # Only the first row of the header file is relevant for the counts; other rows will be ignored.

    return n_vertices, n_cells


def parse_nodes(
    nodes_file: Path,
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Parse Elmer mesh nodes file.
    Args:
        nodes_file (Path): Path to the Elmer mesh nodes file.
    Returns:
        tuple: Global node IDs, x coordinates, y coordinates, and z coordinates.
    """
    df = pd.read_csv(
        nodes_file,
        sep="\\s+",
        header=None,
        dtype={0: int, 2: float, 3: float, 4: float},
    )

    # A valid mesh.nodes file in Elmer should have exactly 5 columns:
    # 1. Global Node ID
    # 2. Node type (usually 0 for regular nodes)
    # 3. x coordinate
    # 4. y coordinate
    # 5. z coordinate
    n_expected_columns: int = 5
    if df.shape[1] < n_expected_columns:
        raise ValueError(
            f"The nodes file has only {df.shape[1]} columns, but it should have exactly {n_expected_columns}."
        )

    global_ids: NDArray[np.int_] = df.iloc[:, 0].to_numpy()
    x: NDArray[np.float64] = df.iloc[:, 2].to_numpy()
    y: NDArray[np.float64] = df.iloc[:, 3].to_numpy()
    z: NDArray[np.float64] = df.iloc[:, 4].to_numpy()

    return global_ids, x, y, z


def parse_elements(elements_file: Path) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Parse Elmer mesh elements file.
    Args:
        elements_file (Path): Path to the Elmer mesh elements file.
    Returns:
        tuple: Element IDs and connectivity.
    """
    df = pd.read_csv(
        elements_file,
        sep="\\s+",
        header=None,
        dtype={0: int, 3: float, 4: float, 5: float},
    )

    # A valid mesh.elements file in Elmer should have at least 2 columns:
    # 1. Global Element ID
    # 2. and 3. will be ignored
    # 4. - 6. column connectivity (list of node IDs that form the triangular element)
    n_expected_columns: int = 6
    if df.shape[1] < n_expected_columns:
        raise ValueError(
            f"The elements file has only {df.shape[1]} columns, but it should have at least {n_expected_columns}."
        )

    global_element_ids: NDArray[np.int_] = df.iloc[:, 0].to_numpy()
    connectivity: NDArray[np.int_] = df.iloc[:, 3:6].to_numpy()

    return global_element_ids, connectivity
