# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

from ebfm.core.cli import parse_cli_args
from ebfm.core.comm import mpi_available
from ebfm.core.config import GridConfig

_ELMER = ["--elmer-mesh-crs-epsg", "3413"]


class TestNumberOfPartitions(unittest.TestCase):
    """The number of partitions of the grid: of the mesh files, or of the split over the ranks."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mesh = Path(self.tmpdir.name)
        for partition_id in (1, 2, 3):
            (self.mesh / f"part.{partition_id}.header").touch()
            (self.mesh / f"part.{partition_id}.nodes").touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    @unittest.skipUnless(mpi_available, "requires MPI (pip install 'ebfm[mpi]')")
    def test_partitioned_mesh(self):
        args = parse_cli_args(
            ["--elmer-mesh", str(self.mesh), *_ELMER, "--is-partitioned-elmer-mesh", "--netcdf-dem-mesh", "dem.nc"]
        )
        args.use_part = 1

        self.assertEqual(GridConfig(args).num_partitions, 3)

    def test_non_partitioned_mesh(self):
        args = parse_cli_args(["--elmer-mesh", str(self.mesh), *_ELMER])

        self.assertEqual(GridConfig(args).num_partitions, 1)

    def test_non_partitioned_elmer_mesh_is_split_over_the_ranks(self):
        args = parse_cli_args(["--elmer-mesh", str(self.mesh), *_ELMER])
        args.use_part = 1

        self.assertEqual(GridConfig(args, num_ranks=4).num_partitions, 4)

    def test_matlab_grid_is_split_over_the_ranks(self):
        args = parse_cli_args(["--matlab-mesh", "mesh.mat"])
        args.use_part = 1

        self.assertEqual(GridConfig(args, num_ranks=4).num_partitions, 4)


if __name__ == "__main__":
    unittest.main()
