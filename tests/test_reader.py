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


def _make_mesh() -> TriangleMesh:
    """Two triangles sharing the edge between the vertices 1 and 2."""
    return TriangleMesh(
        x_vertices=np.array([0.0, 1000.0, 0.0, 1000.0]),
        y_vertices=np.array([-2.0e6, -2.0e6, -2.001e6, -2.001e6]),
        z_vertices=np.zeros(4),
        cell_to_vertex=np.array([[0, 1, 2], [1, 3, 2]]),
        vertex_ids=np.array([1, 2, 3, 4]),
        cell_ids=np.array([1, 2]),
    )


class TestReadDemXios(unittest.TestCase):
    """`read_dem_xios` reads `zs` and `h` per mesh node, laid out as in Elmer/Ice XIOS output (see
    examples/README.md), and averages them per cell."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mesh = _make_mesh()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_dem(self, zs, h) -> Path:
        dem_file = Path(self.tmpdir.name) / "xios_dem.nc"
        with Dataset(dem_file, "w") as nc:
            nc.createDimension("time", None)
            nc.createDimension("nmesh2D_node", len(zs))
            nc.createVariable("zs", "f4", ("time", "nmesh2D_node"))[0, :] = zs
            nc.createVariable("h", "f4", ("time", "nmesh2D_node"))[0, :] = h
        return dem_file

    def test_cell_values_are_means_of_their_vertices(self):
        dem_file = self._write_dem(zs=[100.0, 200.0, 300.0, 400.0], h=[0.0, 3.0, 6.0, 9.0])

        z_cells, h_cells = read_dem_xios(dem_file, self.mesh)

        np.testing.assert_allclose(z_cells, [200.0, 300.0])
        np.testing.assert_allclose(h_cells, [3.0, 6.0])

    def test_vertex_count_mismatch_is_rejected(self):
        dem_file = self._write_dem(zs=[100.0, 200.0, 300.0], h=[0.0, 3.0, 6.0])

        with self.assertRaises(AssertionError):
            read_dem_xios(dem_file, self.mesh)


if __name__ == "__main__":
    unittest.main()
