# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray
import pyproj


def compute_cell_centers_spherical(lon_vertices, lat_vertices, cell_to_vertex):
    """
    Compute cell centers by averaging vertices on the unit sphere.

    Planar triangles (EPSG 3413) become spherical triangles when mapped to lon/lat.

    Uses spherical averaging to find true geometric centers.
    Args:
        lon_vertices: array of vertex longitudes (radians)
        lat_vertices: array of vertex latitudes (radians)
        cell_to_vertex: array mapping spherical triangles (cells) to their local vertex indices
    Returns:
        lon_centers, lat_centers, x_centers, y_centers: arrays of cell center coordinates
    """
    # Convert lon/lat vertices to 3D unit sphere coordinates (Cartesian)
    cart_x = np.cos(lat_vertices) * np.cos(lon_vertices)
    cart_y = np.cos(lat_vertices) * np.sin(lon_vertices)
    cart_z = np.sin(lat_vertices)

    n_cells = cell_to_vertex.shape[0]
    cart_x_centers = np.zeros(n_cells)
    cart_y_centers = np.zeros(n_cells)
    cart_z_centers = np.zeros(n_cells)

    for i in range(n_cells):
        vertex_indices = cell_to_vertex[i]
        cart_x_centers[i] = np.sum(cart_x[vertex_indices])
        cart_y_centers[i] = np.sum(cart_y[vertex_indices])
        cart_z_centers[i] = np.sum(cart_z[vertex_indices])

    norm = np.sqrt(cart_x_centers**2 + cart_y_centers**2 + cart_z_centers**2)
    cart_x_centers /= norm
    cart_y_centers /= norm
    cart_z_centers /= norm

    lon_centers = np.arctan2(cart_y_centers, cart_x_centers)
    lat_centers = np.arcsin(cart_z_centers)

    return lon_centers, lat_centers


class Mesh:
    """A generic 3D Mesh"""

    # x/y/z-coordinates of vertices in a given projection, ordering follows local ids [0,1,...,n_vertices-1]
    x_vertices: NDArray[np.float64]
    y_vertices: NDArray[np.float64]
    # optional height values of vertices stored in z coordinate
    z_vertices: NDArray[np.float64]
    # longitude/latitude coordinates of vertices in radians, ordering follows local ids [0,1,...,n_vertices-1]
    lon_vertices: NDArray[np.float64]
    lat_vertices: NDArray[np.float64]
    # x/y-coordinates of cell centers in a given projection, ordering follows local ids [0,1,...,n_cells-1]
    x_cells: NDArray[np.float64]
    y_cells: NDArray[np.float64]
    # longitude/latitude coordinates of cell centers in radians, ordering follows local ids [0,1,...,n_cells-1]
    lon_cells: NDArray[np.float64]
    lat_cells: NDArray[np.float64]
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
        self.lon_vertices, self.lat_vertices = transformer.transform(self.x_vertices, self.y_vertices, radians=True)

        # Compute cell centers (lon/lat) by averaging vertices on the unit sphere
        self.lon_cells, self.lat_cells = compute_cell_centers_spherical(
            self.lon_vertices, self.lat_vertices, self.cell_to_vertex
        )

        # Convert from LON/LAT to "Polar Stereographic North EPSG 3413"
        inverse_transformer = pyproj.Transformer.from_crs(4326, 3413, always_xy=True)
        self.x_cells, self.y_cells = inverse_transformer.transform(self.lon_cells, self.lat_cells, radians=True)


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
