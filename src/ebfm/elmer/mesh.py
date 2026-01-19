# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray
import pyproj


class Mesh:
    """A generic 3D Mesh"""

    x_vertices: NDArray[np.float64]  # x-coordinates of vertices, ordering follows local ids [0,1,...,n_vertices-1]
    y_vertices: NDArray[np.float64]  # y-coordinates of vertices, ordering follows local ids [0,1,...,n_vertices-1]
    z_vertices: NDArray[np.float64]  # z-coordinates of vertices, ordering follows local ids [0,1,...,n_vertices-1]
    lon: NDArray[
        np.float64
    ]  # longitude coordinates of vertices in radians, ordering follows local ids [0,1,...,n_vertices-1]
    lat: NDArray[
        np.float64
    ]  # latitude coordinates of vertices in radians, ordering follows local ids [0,1,...,n_vertices-1]
    # @TODO later add slope
    # dzdx: NDArray[np.float64]  # z-slope in x-direction
    # dzdy: NDArray[np.float64]  # z-slope in y-direction
    vertex_ids: NDArray[np.int_]  # IDs of vertices, ordering follows local ids [0,1,...,n_vertices-1]
    cell_to_vertex: NDArray[np.int_]  # Mapping from cells to their local vertex IDs
    cell_ids: NDArray[np.int_]  # IDs of cells (triangles), ordering follows local ids [0,1,...,n_cells-1]

    def __init__(
        self,
        x_vertices: NDArray[np.float64],
        y_vertices: NDArray[np.float64],
        z_vertices: NDArray[np.float64],
        cell_to_vertex: NDArray[np.int_],
        vertex_ids: NDArray[np.int_],
        cell_ids: NDArray[np.int_],
    ):
        self.x_vertices = x_vertices
        self.y_vertices = y_vertices
        self.z_vertices = z_vertices
        self.vertex_ids = vertex_ids
        self.cell_ids = cell_ids
        self.cell_to_vertex = cell_to_vertex
        # Convert "Polar Stereographic North EPSG 3413" to LON/LAT (4326)
        transformer = pyproj.Transformer.from_crs(3413, 4326, always_xy=True)
        self.lon, self.lat = transformer.transform(self.x_vertices, self.y_vertices, radians=True)


class TriangleMesh(Mesh):
    """A 3D Mesh consisting of triangular elements."""

    num_vertices_per_cell = 3

    def __init__(
        self,
        x_vertices: NDArray[np.float64],
        y_vertices: NDArray[np.float64],
        z_vertices: NDArray[np.float64],
        cell_to_vertex: NDArray[np.int_],
        vertex_ids: NDArray[np.int_],
        cell_ids: NDArray[np.int_],
    ):
        assert cell_to_vertex.shape[1] == self.num_vertices_per_cell  # a triangle mesh has 3 nodes for all cells
        super(TriangleMesh, self).__init__(x_vertices, y_vertices, z_vertices, cell_to_vertex, vertex_ids, cell_ids)
