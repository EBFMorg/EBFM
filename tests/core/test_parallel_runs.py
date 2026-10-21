# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run of the MATLAB example on several MPI ranks gives exactly the results of a run on a single rank.

EBFM runs once on a single rank and once per rank count under mpirun, and the runs must give byte-identical output
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

REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_MESH = REPO_ROOT / "examples" / "dem_and_mask.mat"
TIMEOUT = 300

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)


# One day: the output file gets its first record after 8 time steps of 3 hours
_TIME = ["--random-seed", "1", "--start-time", "1979-07-01T00:00:00Z", "--end-time", "1979-07-02T00:00:00Z"]
ARGS = ["--matlab-mesh", str(MATLAB_MESH), "--no-shading", *_TIME]


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


def _assert_same_results(serial: Path, parallel: Path) -> None:
    assert filecmp.cmp(serial / "Output" / "model_output.nc", parallel / "Output" / "model_output.nc", shallow=False)

    with np.load(serial / "reference.npz") as expected, np.load(parallel / "reference.npz") as actual:
        assert sorted(actual.files) == sorted(expected.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.fixture(scope="module")
def serial(tmp_path_factory) -> Path:
    return _run(ARGS, 1, tmp_path_factory.mktemp("serial") / "run")


@requires_mpirun
@pytest.mark.parametrize("n_ranks", [2, 3, 4])
def test_parallel_run_equals_serial_run(tmp_path, serial, n_ranks):
    parallel = _run(ARGS, n_ranks, tmp_path / f"np{n_ranks}")

    _assert_same_results(serial, parallel)
