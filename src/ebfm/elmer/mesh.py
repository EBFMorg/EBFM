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
    assert len(lon_vertices) == len(lat_vertices), f"({len(lon_vertices)=} {len(lat_vertices)=})"
    cart_x = np.cos(lat_vertices) * np.cos(lon_vertices)
    cart_y = np.cos(lat_vertices) * np.sin(lon_vertices)
    cart_z = np.sin(lat_vertices)
    assert len(cart_x) == len(cart_y) == len(cart_z), f"({len(cart_x)=} {len(cart_y)=} {len(cart_z)=})"

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
    # EPSG code of the projection of x/y
    source_crs_epsg: int
    # The following coordinates are derived from x/y_vertices by compute_geometry()
    # longitude/latitude coordinates of vertices in radians, ordering follows local ids [0,1,...,n_vertices-1]
    lon_vertices: NDArray[np.float64]
    lat_vertices: NDArray[np.float64]
    # x/y-coordinates of cell centers in a given projection, ordering follows local ids [0,1,...,n_cells-1]
    x_cells: NDArray[np.float64]
    y_cells: NDArray[np.float64]
    z_cells: NDArray[np.float64]
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
        source_crs_epsg: int = 3413,
    ):
        self.x_vertices = x_vertices
        self.y_vertices = y_vertices
        self.z_vertices = z_vertices
        self.vertex_ids = vertex_ids
        self.cell_ids = cell_ids
        self.cell_to_vertex = cell_to_vertex
        self.source_crs_epsg = source_crs_epsg
        # Initialize z_cells to zeros (to be filled/overwritten by DEM)
        n_cells = self.cell_to_vertex.shape[0]
        self.z_cells = np.zeros(n_cells)

    def compute_geometry(self) -> None:
        """
        Derive the lon/lat coordinates of the vertices and the coordinates of the cell centers.

        Transforming the vertices is the expensive part of setting up a mesh. Call this once the mesh holds the cells
        it will be used with, e.g. after get_partition, so that only those are transformed.
        """
        transformer = pyproj.Transformer.from_crs(self.source_crs_epsg, 4326, always_xy=True)
        self.lon_vertices, self.lat_vertices = transformer.transform(self.x_vertices, self.y_vertices, radians=True)

        # Compute cell centers (lon/lat) by averaging vertices on the unit sphere
        self.lon_cells, self.lat_cells = compute_cell_centers_spherical(
            self.lon_vertices, self.lat_vertices, self.cell_to_vertex
        )

        inverse_transformer = pyproj.Transformer.from_crs(4326, self.source_crs_epsg, always_xy=True)
        self.x_cells, self.y_cells = inverse_transformer.transform(self.lon_cells, self.lat_cells, radians=True)

    def get_partition(self, partition_id: int, n_partitions: int) -> "Mesh":
        """
        The part of this mesh that one of `n_partitions` partitions holds.

        The cells are split into `n_partitions` contiguous blocks in their current order, and partition 1 holds the
        first block. A partition keeps the vertices its cells use, so a vertex on the border between two partitions
        belongs to both.

        @param[in] partition_id id of the partition, from 1 to n_partitions
        @param[in] n_partitions number of partitions

        @returns a mesh of the same type with the cells of the partition, including their z_cells
        """
        assert 1 <= partition_id <= n_partitions, f"Partition id {partition_id} is not in 1..{n_partitions}."

        cells = np.array_split(np.arange(len(self.cell_ids)), n_partitions)[partition_id - 1]
        vertices, cell_to_vertex = np.unique(self.cell_to_vertex[cells], return_inverse=True)

        partition = type(self)(
            x_vertices=self.x_vertices[vertices],
            y_vertices=self.y_vertices[vertices],
            z_vertices=self.z_vertices[vertices],
            cell_to_vertex=cell_to_vertex.reshape(len(cells), -1),
            vertex_ids=self.vertex_ids[vertices],
            cell_ids=self.cell_ids[cells],
            source_crs_epsg=self.source_crs_epsg,
        )
        partition.z_cells = self.z_cells[cells]
        return partition


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
        source_crs_epsg: int = 3413,
    ):
        assert cell_to_vertex.shape[1] == self.num_vertices_per_cell  # a triangle mesh has 3 nodes for all cells
        super().__init__(
            x_vertices,
            y_vertices,
            z_vertices,
            cell_to_vertex,
            vertex_ids,
            cell_ids,
            source_crs_epsg,
        )
