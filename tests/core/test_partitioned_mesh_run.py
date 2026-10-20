# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""The results of EBFM on a partitioned Elmer mesh do not depend on the partitioning.

A serial run on the whole mesh and a run on 3 MPI ranks, one partition each, must give equal --dump-reference
snapshots. The mesh spans 70 degrees north, a bin boundary of the water vapour coefficient in LOOP_EBM_SWin, and its
elevation rises non-linearly from south to north, so the partitions, latitude bands, differ in their latitude mean
and in the lapse rate fitted over their own columns.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

TIMEOUT = 600
N_PARTITIONS = 3

# EPSG:3413 coordinates of the mesh, latitudes 67.6 to 71.2 degrees north
X0, Y0 = -100000.0, -2450000.0
NX, NY = 4, 10
SPACING = 40000.0

_ARGS = [
    "--netcdf-dem-mesh",
    "dem.nc",
    "--elmer-mesh-crs-epsg",
    "3413",
    "--random-seed",
    "1",
    "--start-time",
    "1979-07-01T00:00:00Z",
    "--end-time",
    "1979-07-01T12:00:00Z",
    "--dump-reference",
    "reference.npz",
]

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)


def _write_mesh(mesh_dir: Path) -> None:
    """
    Write a rectangle of NX x NY squares, two triangles each, as an Elmer mesh and as N_PARTITIONS partitions of it.

    Each partition is a contiguous block of elements, holding the nodes its elements use, so the element ids of the
    partitions, in rank order, are those of the whole mesh.
    """
    mesh_dir.mkdir()
    node_id = np.arange(1, (NX + 1) * (NY + 1) + 1).reshape(NY + 1, NX + 1)
    nodes = {
        node_id[j, i]: f"{node_id[j, i]} -1 {X0 + i * SPACING} {Y0 + j * SPACING} 0.0\n"
        for j in range(NY + 1)
        for i in range(NX + 1)
    }

    triangles = []
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = node_id[j, i], node_id[j, i + 1], node_id[j + 1, i + 1], node_id[j + 1, i]
            triangles += [(a, b, c), (a, c, d)]
    elements = [(element_id, triangle) for element_id, triangle in enumerate(triangles, start=1)]

    def write(prefix: str, part_elements) -> None:
        part_nodes = sorted({node for _, triangle in part_elements for node in triangle})
        (mesh_dir / f"{prefix}.header").write_text(f"{len(part_nodes)} {len(part_elements)} 0\n")
        (mesh_dir / f"{prefix}.nodes").write_text("".join(nodes[node] for node in part_nodes))
        (mesh_dir / f"{prefix}.elements").write_text(
            "".join(f"{element_id} 1 303 {a} {b} {c}\n" for element_id, (a, b, c) in part_elements)
        )

    write("mesh", elements)
    for partition_id, part_elements in enumerate(np.array_split(np.arange(len(elements)), N_PARTITIONS), start=1):
        write(f"part.{partition_id}", [elements[k] for k in part_elements])


def _write_dem(dem_file: Path) -> None:
    """Write a DEM for --netcdf-dem-mesh whose surface rises quadratically from 0 m in the south to 2000 m."""
    x = X0 + SPACING / 4 * np.arange(-1, 4 * NX + 2)
    y = Y0 + SPACING / 4 * np.arange(-1, 4 * NY + 2)
    surface = 2000.0 * ((y - y[0]) / (y[-1] - y[0])) ** 2

    with Dataset(dem_file, "w") as nc:
        nc.createDimension("x", len(x))
        nc.createDimension("y", len(y))
        nc.createVariable("x", "f8", ("x",))[:] = x
        nc.createVariable("y", "f8", ("y",))[:] = y
        nc.createVariable("surface", "f8", ("y", "x"))[:] = np.repeat(surface[:, np.newaxis], len(x), axis=1)


def _run(workdir: Path, mesh_dir: Path, n_ranks: int, *args: str) -> dict[str, np.ndarray]:
    """Run EBFM on n_ranks ranks in workdir; return its --dump-reference snapshot."""
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = [sys.executable, "-m", "ebfm.main", "--elmer-mesh", str(mesh_dir), *_ARGS, *args]
    if n_ranks > 1:
        cmd = ["mpirun", "-np", str(n_ranks), *cmd]

    result = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"
    with np.load(workdir / "reference.npz") as reference:
        return dict(reference)


@requires_mpirun
def test_partitioned_run_equals_serial_run(tmp_path):
    mesh_dir = tmp_path / "MESH"
    _write_mesh(mesh_dir)
    _write_dem(tmp_path / "dem.nc")

    serial = _run(tmp_path, mesh_dir, 1)
    partitioned = _run(tmp_path, mesh_dir, N_PARTITIONS, "--is-partitioned-elmer-mesh")

    assert sorted(partitioned) == sorted(serial)
    for key in serial:
        np.testing.assert_array_equal(partitioned[key], serial[key], err_msg=key)
