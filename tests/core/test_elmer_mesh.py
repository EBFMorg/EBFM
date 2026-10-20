# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np

from ebfm.elmer.mesh import TriangleMesh

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


if __name__ == "__main__":
    unittest.main()
