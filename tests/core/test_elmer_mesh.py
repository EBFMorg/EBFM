# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ebfm.elmer.mesh import TriangleMesh
from ebfm.reader import read_dem_xios

N_CELLS = 10
N_PARTITIONS = 3


def _strip_mesh() -> TriangleMesh:
    """A strip of N_CELLS triangles in EPSG:3413, cell i spanning the vertices i, i+1 and i+2.

    Like in an Elmer mesh, vertex and cell ids are global and neither start at 0 nor are contiguous.
    """
    n_vertices = N_CELLS + 2
    mesh = TriangleMesh(
        x_vertices=1000.0 * np.arange(n_vertices),
        y_vertices=-2.0e6 + 1000.0 * (np.arange(n_vertices) % 2),
        z_vertices=np.zeros(n_vertices),
        cell_to_vertex=np.array([[i, i + 1, i + 2] for i in range(N_CELLS)]),
        vertex_ids=10 + 3 * np.arange(n_vertices),
        cell_ids=7 + 5 * np.arange(N_CELLS),
    )
    mesh.z_cells = np.arange(N_CELLS, dtype=np.float64)
    return mesh


class TestGetPartition(unittest.TestCase):
    """Splitting a mesh into partitions keeps every cell, in order, with its geometry and its vertices."""

    def setUp(self):
        self.mesh = _strip_mesh()
        self.partitions = [self.mesh.get_partition(p, N_PARTITIONS) for p in range(1, N_PARTITIONS + 1)]
        for mesh in (self.mesh, *self.partitions):
            mesh.compute_geometry()

    def test_partitions_hold_contiguous_blocks_with_the_first_block_in_partition_1(self):
        self.assertEqual([len(p.cell_ids) for p in self.partitions], [4, 3, 3])
        np.testing.assert_array_equal(np.concatenate([p.cell_ids for p in self.partitions]), self.mesh.cell_ids)

    def test_cell_centres_and_elevations_equal_those_of_the_whole_mesh(self):
        for name in ("x_cells", "y_cells", "lon_cells", "lat_cells", "z_cells"):
            with self.subTest(name=name):
                combined = np.concatenate([getattr(p, name) for p in self.partitions])
                np.testing.assert_array_equal(combined, getattr(self.mesh, name))

    def test_cells_keep_their_vertices(self):
        whole = self.mesh.vertex_ids[self.mesh.cell_to_vertex]
        combined = np.concatenate([p.vertex_ids[p.cell_to_vertex] for p in self.partitions])
        np.testing.assert_array_equal(combined, whole)

    def test_partition_holds_only_the_vertices_of_its_cells(self):
        for partition in self.partitions:
            with self.subTest(first_cell=partition.cell_ids[0]):
                self.assertEqual(len(partition.vertex_ids), len(np.unique(partition.cell_to_vertex)))
                self.assertEqual(len(partition.vertex_ids), len(partition.x_vertices))

    def test_partition_id_out_of_range_is_rejected(self):
        for partition_id in (0, N_PARTITIONS + 1):
            with self.subTest(partition_id=partition_id), self.assertRaises(AssertionError):
                self.mesh.get_partition(partition_id, N_PARTITIONS)


def _write_xios_dem(path: Path, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Write zs and h for node ids 1..n_nodes, stored by node id as Elmer/Ice XIOS output does; return them."""
    node_ids = np.arange(1, n_nodes + 1)
    zs = (500.0 + 7.0 * node_ids).astype(np.float32)
    h = np.where(node_ids <= n_nodes // 3, 0.5, 100.0 + node_ids).astype(np.float32)
    with Dataset(path, "w") as nc:
        nc.createDimension("time", None)
        nc.createDimension("nmesh2D_node", n_nodes)
        for name, values in (("zs", zs), ("h", h)):
            nc.createVariable(name, "f4", ("time", "nmesh2D_node"))[0, :] = values
    return zs, h


def _renumbered(mesh: TriangleMesh) -> TriangleMesh:
    """The same mesh with its vertices listed in reverse order, as in a mesh.nodes not sorted by node id."""
    order = np.arange(len(mesh.vertex_ids))[::-1]
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    return TriangleMesh(
        x_vertices=mesh.x_vertices[order],
        y_vertices=mesh.y_vertices[order],
        z_vertices=mesh.z_vertices[order],
        cell_to_vertex=position[mesh.cell_to_vertex],
        vertex_ids=mesh.vertex_ids[order],
        cell_ids=mesh.cell_ids,
    )


class TestReadDemXiosByNodeId(unittest.TestCase):
    """read_dem_xios looks up the node values by Elmer node id, so any part or order of the mesh gets the same cells."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dem_file = Path(self.tmpdir.name) / "xios_dem.nc"
        self.mesh = _strip_mesh()
        self.zs, self.h = _write_xios_dem(self.dem_file, n_nodes=self.mesh.vertex_ids.max())

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cells_get_the_mean_of_their_corner_nodes(self):
        z, h = read_dem_xios(self.dem_file, self.mesh)

        corner_ids = self.mesh.vertex_ids[self.mesh.cell_to_vertex]
        np.testing.assert_allclose(z, self.zs[corner_ids - 1].mean(axis=1), rtol=1e-6)
        np.testing.assert_allclose(h, self.h[corner_ids - 1].mean(axis=1), rtol=1e-6)

    def test_renumbered_mesh_gets_the_same_cells(self):
        expected = read_dem_xios(self.dem_file, self.mesh)

        for actual, wanted in zip(read_dem_xios(self.dem_file, _renumbered(self.mesh)), expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_partitions_get_the_cells_of_the_whole_mesh(self):
        expected = read_dem_xios(self.dem_file, self.mesh)

        parts = [read_dem_xios(self.dem_file, self.mesh.get_partition(p, N_PARTITIONS)) for p in (1, 2, 3)]
        for field, wanted in enumerate(expected):
            np.testing.assert_array_equal(np.concatenate([part[field] for part in parts]), wanted)


if __name__ == "__main__":
    unittest.main()
