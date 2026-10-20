# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Output of columns distributed over several MPI ranks is gathered on one rank and written as by a serial run."""

import filecmp
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ebfm.core.comm import defaultComm
from ebfm.core.comm.gather import ColumnGather
from test_loop_write_to_file import _structured_grid, _unstructured_grid_with_mesh, _write_single_step

COMM_TESTS = Path(__file__).resolve().parent / "comm"
TIMEOUT = 120

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)


def _mpirun(n_ranks: int, script: str, *args: str) -> subprocess.CompletedProcess:
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = ["mpirun", "-np", str(n_ranks), sys.executable, str(COMM_TESTS / script), *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT)


def test_single_rank_gathers_without_communication():
    column_ids = np.array([4, 2, 9])
    values = np.array([1.0, 2.0, 3.0])

    gather = ColumnGather(defaultComm, column_ids)

    assert gather.is_root
    assert gather.gather(values) is values
    assert gather.column_ids is column_ids


@requires_mpirun
def test_gather_on_three_ranks():
    result = _mpirun(3, "gather_columns.py")

    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(3):
        assert f"rank {rank}: ok" in result.stdout, result.stdout + result.stderr


@requires_mpirun
@pytest.mark.parametrize("n_ranks", [2, 3])
@pytest.mark.parametrize(
    "grid_kind, make_grid",
    [("structured", _structured_grid), ("unstructured", _unstructured_grid_with_mesh)],
)
def test_parallel_output_file_equals_serial_one(tmp_path, n_ranks, grid_kind, make_grid):
    serial_file, _, _ = _write_single_step(tmp_path / "serial", make_grid())
    result = _mpirun(n_ranks, "write_output_on_ranks.py", grid_kind, str(tmp_path / "parallel"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert filecmp.cmp(serial_file, tmp_path / "parallel" / "model_output.nc", shallow=False)
