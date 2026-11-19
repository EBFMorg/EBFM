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
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def partitioning_dir(mesh_dir: Path, n_partitions: int) -> Path:
    """The directory of the partitioning of the Elmer mesh in mesh_dir into n_partitions, as ElmerGrid names it."""
    return Path(mesh_dir) / f"partitioning.{n_partitions}"


def partition_elmer_mesh(mesh_dir: Path, n_partitions: int, out_dir: Path | None = None) -> Path:
    """
    Write n_partitions partitions of the Elmer mesh in mesh_dir to out_dir, partitioning_dir() by default.

    The partitions are written to a temporary directory next to out_dir, which then takes its place, so that out_dir
    never holds a partial partitioning, e.g. after an interrupted run or while another run writes the same one.

    @returns out_dir
    @raises FileExistsError if out_dir exists
    """
    out_dir = Path(out_dir) if out_dir is not None else partitioning_dir(mesh_dir, n_partitions)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists.")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        _write_partitions(Path(mesh_dir), n_partitions, tmp_dir)
        os.rename(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return out_dir


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
