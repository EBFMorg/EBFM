# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import tempfile
import unittest
from pathlib import Path

from ebfm.grid.elmer.partitioning import partition_elmer_mesh, partitioning_dir

ELMER_MESH = Path(__file__).parents[2] / "data" / "elmer_mesh" / "MESH"


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


class TestPartitionElmerMesh(unittest.TestCase):
    """The partitions of an Elmer mesh are contiguous blocks of its elements, with its node lines unchanged."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.mesh = Path(self.tmpdir.name) / "MESH"
        shutil.copytree(ELMER_MESH, self.mesh)
        self.partitioning = partition_elmer_mesh(self.mesh, 3)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_written_to_the_partitioning_directory_of_elmergrid(self):
        self.assertEqual(self.partitioning, partitioning_dir(self.mesh, 3))
        self.assertEqual(self.partitioning, self.mesh / "partitioning.3")

    def test_partitions_hold_the_elements_in_order(self):
        elements = [_lines(self.partitioning / f"part.{i}.elements") for i in (1, 2, 3)]

        self.assertEqual(sum(elements, []), _lines(self.mesh / "mesh.elements"))
        self.assertLessEqual(max(map(len, elements)) - min(map(len, elements)), 1)

    def test_partitions_hold_the_node_lines_of_their_elements(self):
        whole_nodes = {line.split()[0]: line for line in _lines(self.mesh / "mesh.nodes")}
        for i in (1, 2, 3):
            with self.subTest(partition=i):
                nodes = _lines(self.partitioning / f"part.{i}.nodes")
                used = {node for line in _lines(self.partitioning / f"part.{i}.elements") for node in line.split()[3:]}
                self.assertEqual({line.split()[0] for line in nodes}, used)
                self.assertEqual(nodes, [whole_nodes[line.split()[0]] for line in nodes])
                header = _lines(self.partitioning / f"part.{i}.header")[0].split()
                self.assertEqual(
                    header[:2], [str(len(nodes)), str(len(_lines(self.partitioning / f"part.{i}.elements")))]
                )

    def test_existing_partitioning_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            partition_elmer_mesh(self.mesh, 3)
        self.assertEqual(sorted(p.name for p in self.mesh.iterdir() if p.name.startswith(".")), [])


if __name__ == "__main__":
    unittest.main()
