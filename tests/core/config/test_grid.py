# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

from ebfm.core.cli import parse_cli_args
from ebfm.core.config import GridConfig

_ELMER = ["--elmer-mesh-crs-epsg", "3413"]


class TestNumberOfPartitions(unittest.TestCase):
    """The number of partitions of the grid: of the mesh files, or of the partitioning of a structured grid."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mesh = Path(self.tmpdir.name)
        for partition_id in (1, 2, 3):
            (self.mesh / f"part.{partition_id}.header").touch()
            (self.mesh / f"part.{partition_id}.nodes").touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_partitioned_mesh(self):
        args = parse_cli_args(["--elmer-mesh", str(self.mesh.parent), *_ELMER])
        args.partitioning, args.partition_id = self.mesh, 1

        grid_config = GridConfig(args)

        self.assertEqual(grid_config.num_partitions, 3)
        self.assertEqual(grid_config.mesh_file, self.mesh)

    def test_non_partitioned_mesh(self):
        args = parse_cli_args(["--elmer-mesh", str(self.mesh), *_ELMER])

        self.assertEqual(GridConfig(args).num_partitions, 1)

    def test_partition_of_a_structured_grid_is_read_from_its_partitioning(self):
        for grid_args, suffix in ((["--matlab-mesh", "grid.mat"], ".mat"), (["--netcdf-mesh", "grid.nc"], ".nc")):
            with self.subTest(grid_args[0]):
                partitioning = Path(self.tmpdir.name) / f"grid{suffix}.partitioning.4"
                partitioning.mkdir()
                for partition_id in range(1, 5):
                    (partitioning / f"part.{partition_id}{suffix}").touch()
                args = parse_cli_args(grid_args)
                args.partitioning, args.partition_id = partitioning, 3

                grid_config = GridConfig(args)

                self.assertTrue(grid_config.is_partitioned)
                self.assertEqual(grid_config.num_partitions, 4)
                self.assertEqual(grid_config.partition_id, 3)
                self.assertEqual(grid_config.mesh_file, partitioning / f"part.3{suffix}")


if __name__ == "__main__":
    unittest.main()
