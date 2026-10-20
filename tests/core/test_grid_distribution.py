# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every column of the domain must be held by exactly one MPI rank, and reductions run over all of them."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ebfm.core.comm import defaultComm
from ebfm.core.grid import domain_mean, validate_grid_distribution

COMM_TESTS = Path(__file__).resolve().parent / "comm"
TIMEOUT = 120

requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="mpirun or mpi4py not available",
)


def test_single_rank_holds_every_column():
    validate_grid_distribution({"column_id": np.array([4, 2, 9])}, defaultComm)  # must not raise


@requires_mpirun
@pytest.mark.parametrize(
    "mode, expected",
    [("distinct", "ok"), ("duplicate", "Some columns are held by more than one of the 3 MPI ranks")],
)
def test_three_ranks(mode, expected):
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = ["mpirun", "-np", "3", sys.executable, str(COMM_TESTS / "validate_grid_distribution.py"), mode]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT)

    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(3):
        assert f"rank {rank}: {expected}" in result.stdout, result.stdout + result.stderr


def test_domain_mean_on_a_single_rank_is_the_serial_mean():
    values = np.array([61.3, 78.2, 80.1, 72.9])

    assert domain_mean(values, defaultComm) == np.mean(values)


@requires_mpirun
def test_domain_mean_on_three_ranks():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    cmd = ["mpirun", "-np", "3", sys.executable, str(COMM_TESTS / "domain_mean.py")]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT)

    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(3):
        assert f"rank {rank}: ok" in result.stdout, result.stdout + result.stderr
