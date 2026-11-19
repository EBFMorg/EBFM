# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split an Elmer mesh into partitions that EBFM can run on, one MPI rank per partition.

Writes <mesh_dir>/partitioning.<n>/part.<i>.{header,nodes,elements} for i = 1..n. Partition i holds the i-th of n
contiguous blocks of the elements, in the order of mesh.elements, and the nodes its elements use. Node and element
lines are copied unchanged, so the partitions carry exactly the coordinates of the whole mesh.

On n MPI ranks, EBFM runs on <mesh_dir>/partitioning.<n> when given --elmer-mesh <mesh_dir>, and creates it with this
module if it is missing and --create-partitioning is given. To create it beforehand:

    python -m ebfm.grid.elmer.partitioning MESH 4
    mpirun -np 4 ebfm --elmer-mesh MESH ...

The partitions are not those of ElmerGrid and carry no boundary or shared-node files, so Elmer/Ice cannot run on
them. When coupling to Elmer/Ice, partition the mesh with ElmerGrid instead.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ebfm.grid.partitioning import write_directory_atomically


def partitioning_dir(mesh_dir: Path, n_partitions: int) -> Path:
    """The directory of the partitioning of the Elmer mesh in mesh_dir into n_partitions, as ElmerGrid names it."""
    return Path(mesh_dir) / f"partitioning.{n_partitions}"


def partition_column_ids(partitioning: Path) -> dict[int, np.ndarray]:
    """The column ids, i.e. element ids, of every partition of a partitioning of an Elmer mesh, by partition id."""
    element_files = {int(f.name.split(".")[1]): f for f in Path(partitioning).glob("part.*.elements")}
    if not element_files or sorted(element_files) != list(range(1, len(element_files) + 1)):
        raise ValueError(f"{partitioning} holds no partitions part.<i>.elements numbered from 1.")
    return {
        partition_id: np.loadtxt(element_files[partition_id], usecols=0, dtype=np.int64, ndmin=1)
        for partition_id in sorted(element_files)
    }


def partition_elmer_mesh(mesh_dir: Path, n_partitions: int, out_dir: Path | None = None) -> Path:
    """
    Write n_partitions partitions of the Elmer mesh in mesh_dir to out_dir, partitioning_dir() by default.

    The partitions are written atomically (see ebfm.grid.partitioning.write_directory_atomically).

    @returns out_dir
    @raises FileExistsError if out_dir exists
    """
    out_dir = Path(out_dir) if out_dir is not None else partitioning_dir(mesh_dir, n_partitions)
    return write_directory_atomically(out_dir, lambda tmp_dir: _write_partitions(Path(mesh_dir), n_partitions, tmp_dir))


def _write_partitions(mesh_dir: Path, n_partitions: int, out_dir: Path) -> None:
    """Write n_partitions partitions of the Elmer mesh in mesh_dir to the existing directory out_dir."""
    if n_partitions < 1:
        raise ValueError(f"The number of partitions must be at least 1, not {n_partitions}.")

    element_lines = [line for line in (mesh_dir / "mesh.elements").read_text().splitlines() if line.strip()]
    if len(element_lines) < n_partitions:
        raise ValueError(f"{mesh_dir} has {len(element_lines)} elements, too few for {n_partitions} partitions.")
    node_lines = {}
    for line in (mesh_dir / "mesh.nodes").read_text().splitlines():
        if line.strip():
            node_lines[int(line.split()[0])] = line

    for partition_id, block in enumerate(np.array_split(np.arange(len(element_lines)), n_partitions), start=1):
        lines = [element_lines[k] for k in block]
        # An element line is: element id, body id, element type, node ids
        node_ids = sorted({int(node_id) for line in lines for node_id in line.split()[3:]})
        type_counts = Counter(line.split()[2] for line in lines)

        header = [f"{len(node_ids)} {len(lines)} 0", str(len(type_counts))]
        header += [f"{element_type} {count}" for element_type, count in sorted(type_counts.items())]
        (out_dir / f"part.{partition_id}.header").write_text("\n".join(header) + "\n")
        (out_dir / f"part.{partition_id}.nodes").write_text("".join(node_lines[node_id] + "\n" for node_id in node_ids))
        (out_dir / f"part.{partition_id}.elements").write_text("".join(line + "\n" for line in lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh_dir", type=Path, help="directory of the Elmer mesh (mesh.header, mesh.nodes, ...)")
    parser.add_argument("n_partitions", type=int, help="number of partitions, i.e. of MPI ranks to run EBFM on")
    parser.add_argument(
        "-o", "--output", type=Path, help="directory to write the partitions to (default: MESH/partitioning.<n>)"
    )
    args = parser.parse_args(argv)

    try:
        out_dir = partition_elmer_mesh(args.mesh_dir, args.n_partitions, args.output)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {args.n_partitions} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
