# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run on several MPI ranks gives exactly the results of a run on a single rank.

Each case runs EBFM once on a single rank and once per rank count under mpirun, on the partitioning of the grid that
EBFM creates with --create-partitioning, and requires equal --dump-reference snapshots and equal output, once the
output files of the partitions are merged by ebfm.io.merge.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_MESH = REPO_ROOT / "examples" / "dem_and_mask.mat"
ELMER_MESH = REPO_ROOT / "tests" / "data" / "elmer_mesh" / "MESH"
XIOS_DEM = ELMER_MESH.parent / "xios_dem.nc"
TIMEOUT = 300

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)


# One day: the output file gets its first record after 8 time steps of 3 hours
_TIME = ["--random-seed", "1", "--start-time", "1979-07-01T00:00:00Z", "--end-time", "1979-07-02T00:00:00Z"]
_ELMER = ["--elmer-mesh", str(ELMER_MESH), "--elmer-mesh-crs-epsg", "3413"]

CASES = {
    "matlab": ["--matlab-mesh", str(MATLAB_MESH), "--no-shading", *_TIME],
    "elmer": [*_ELMER, *_TIME],
    "elmerxios": [*_ELMER, "--netcdf-dem-mesh-unstructured", str(XIOS_DEM), *_TIME],
    "fake-coupling": [
        *_ELMER,
        "--couple-to-elmer-ice",
        "--couple-to-icon-atmo",
        "--couple-to-icon-land",
        "--fake-coupling",
        *_TIME,
    ],
}


def _run(args: list[str], n_ranks: int, workdir: Path) -> Path:
    """Run EBFM on n_ranks ranks in workdir, with --dump-reference; return workdir."""
    workdir.mkdir(parents=True)
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = [sys.executable, "-m", "ebfm.main", *args, "--dump-reference", "reference.npz", "--log-file", "ebfm.log"]
    if n_ranks > 1:
        cmd = ["mpirun", "-np", str(n_ranks), *cmd]

    try:
        result = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{' '.join(cmd)} timed out after {TIMEOUT} s. Last log lines per rank:\n{_log_tails(workdir)}")
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"
    return workdir


def _log_tails(workdir: Path, n_lines: int = 15) -> str:
    """The last lines of the log file of every rank in workdir."""
    tails = []
    for log_file in sorted(workdir.glob("ebfm.log*")):
        lines = log_file.read_text(errors="replace").splitlines()[-n_lines:]
        tails.append(f"--- {log_file.name}\n" + "\n".join(lines))
    return "\n".join(tails) or "(no log files)"


def tool(module: str, *args) -> None:
    """Run a module of ebfm.io standalone."""
    cmd = [sys.executable, "-m", f"ebfm.io.{module}", *map(str, args)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}"


def parallel_args(case: str, tmp_path: Path) -> list[str]:
    """
    The arguments of case for a run on several ranks: on a copy of its grid in tmp_path, whose partitioning for the
    number of ranks EBFM creates with --create-partitioning if it does not exist yet.
    """
    args = list(CASES[case])
    for option, source in (("--elmer-mesh", ELMER_MESH), ("--matlab-mesh", MATLAB_MESH)):
        if option in args:
            copy = tmp_path / "grid" / source.name
            if not copy.exists():
                copy.parent.mkdir(exist_ok=True)
                (shutil.copytree if source.is_dir() else shutil.copy)(source, copy)
            args[args.index(option) + 1] = str(copy)
    return [*args, "--create-partitioning"]


def assert_same_variables(expected_file: Path, actual_file: Path) -> None:
    """Every variable of expected_file is in actual_file, with the same values."""
    with Dataset(expected_file) as expected, Dataset(actual_file) as actual:
        for name, var in expected.variables.items():
            np.testing.assert_array_equal(actual[name][:], var[:], err_msg=name)


def _assert_same_results(serial: Path, parallel: Path) -> None:
    (output_file,) = (serial / "Output").glob("output_*.nc")
    merged = parallel / "merged.nc"
    tool("merge", parallel / "Output" / output_file.stem, "-o", merged)
    assert_same_variables(output_file, merged)

    with np.load(serial / "reference.npz") as expected, np.load(parallel / "reference.npz") as actual:
        assert sorted(actual.files) == sorted(expected.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@requires_mpirun
@pytest.mark.parametrize(
    "case, n_ranks",
    [
        ("matlab", 2),
        ("matlab", 3),
        ("matlab", 4),
        ("elmer", 3),
        ("elmerxios", 3),
        ("fake-coupling", 2),
    ],
    ids=str,
)
def test_parallel_run_equals_serial_run(tmp_path, case, n_ranks):
    serial = _run(CASES[case], 1, tmp_path / "serial")
    parallel = _run(parallel_args(case, tmp_path), n_ranks, tmp_path / f"np{n_ranks}")

    _assert_same_results(serial, parallel)
