# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run on several MPI ranks gives exactly the results of a run on a single rank.

Each case runs EBFM once on a single rank and once per rank count under mpirun, and requires byte-identical output
files and equal --dump-reference snapshots.
"""

import filecmp
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_elmer_mesh import _write_xios_dem

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TIMEOUT = 600

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)

# One day: the output file gets its first record after 8 time steps of 3 hours
_TIME = ["--random-seed", "1", "--end-time", "1979-07-02T00:00:00Z", "--start-time", "1979-07-01T00:00:00Z"]


def _write_elmer_mesh(mesh_dir: Path, nx: int = 12, ny: int = 8, spacing: float = 5000.0) -> None:
    """Write a rectangle of nx x ny squares, two triangles each, in EPSG:3413 as an Elmer mesh."""
    mesh_dir.mkdir(parents=True)
    node_id = np.arange(1, (nx + 1) * (ny + 1) + 1).reshape(ny + 1, nx + 1)
    x0, y0 = -200000.0, -2000000.0

    with open(mesh_dir / "mesh.nodes", "w") as nodes:
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.write(f"{node_id[j, i]} -1 {x0 + i * spacing} {y0 + j * spacing} 0.0\n")

    triangles = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = node_id[j, i], node_id[j, i + 1], node_id[j + 1, i + 1], node_id[j + 1, i]
            triangles += [(a, b, c), (a, c, d)]
    with open(mesh_dir / "mesh.elements", "w") as elements:
        for element_id, (a, b, c) in enumerate(triangles, start=1):
            elements.write(f"{element_id} 1 303 {a} {b} {c}\n")

    (mesh_dir / "mesh.header").write_text(f"{node_id.size} {len(triangles)} 0\n")


def _run(args: list[str], n_ranks: int, workdir: Path) -> Path:
    """Run EBFM on n_ranks ranks in workdir, with --dump-reference; return workdir."""
    workdir.mkdir(parents=True)
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = [sys.executable, "-m", "ebfm.main", *args, "--dump-reference", "reference.npz"]
    if n_ranks > 1:
        cmd = ["mpirun", "-np", str(n_ranks), *cmd]

    result = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"
    return workdir


def _assert_same_results(serial: Path, parallel: Path) -> None:
    assert filecmp.cmp(serial / "Output" / "model_output.nc", parallel / "Output" / "model_output.nc", shallow=False)

    with np.load(serial / "reference.npz") as expected, np.load(parallel / "reference.npz") as actual:
        assert sorted(actual.files) == sorted(expected.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.fixture(scope="module")
def elmer_mesh(tmp_path_factory) -> Path:
    mesh_dir = tmp_path_factory.mktemp("elmer") / "MESH"
    _write_elmer_mesh(mesh_dir)
    n_nodes = int((mesh_dir / "mesh.header").read_text().split()[0])
    _write_xios_dem(mesh_dir.parent / "xios_dem.nc", n_nodes)
    return mesh_dir


CASES = {
    "matlab": lambda mesh: ["--matlab-mesh", str(EXAMPLES / "dem_and_mask.mat"), "--no-shading", *_TIME],
    "elmer": lambda mesh: ["--elmer-mesh", str(mesh), "--elmer-mesh-crs-epsg", "3413", *_TIME],
    "elmerxios": lambda mesh: [
        "--elmer-mesh",
        str(mesh),
        "--elmer-mesh-crs-epsg",
        "3413",
        "--netcdf-dem-mesh-unstructured",
        str(mesh.parent / "xios_dem.nc"),
        *_TIME,
    ],
    "fake-coupling": lambda mesh: [
        "--elmer-mesh",
        str(mesh),
        "--elmer-mesh-crs-epsg",
        "3413",
        "--couple-to-elmer-ice",
        "--couple-to-icon-atmo",
        "--couple-to-icon-land",
        "--fake-coupling",
        *_TIME,
    ],
}


@requires_mpirun
@pytest.mark.parametrize(
    "case, n_ranks",
    [("matlab", 3), ("matlab", 4), ("elmer", 3), ("elmerxios", 3), ("fake-coupling", 2)],
    ids=lambda value: str(value),
)
def test_parallel_run_equals_serial_run(tmp_path, elmer_mesh, case, n_ranks):
    args = CASES[case](elmer_mesh)

    serial = _run(args, 1, tmp_path / "serial")
    parallel = _run(args, n_ranks, tmp_path / f"np{n_ranks}")

    _assert_same_results(serial, parallel)
