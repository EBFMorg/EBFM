# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fatal error must end EBFM with a non-zero exit status, standalone and under mpirun, and end an MPMD run."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATE_FATAL_ERROR = REPO_ROOT / "tests" / "core" / "comm" / "simulate_fatal_error.py"
STAGES = ["after-comm-splitting", "before-comm-splitting"]
TIMEOUT = 120


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # Once mpi4py has initialised MPI in this process, the process environment holds the launcher variables of an
    # MPI singleton, and an mpirun inheriting them exits 1 without running anything.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_"))}
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT)


def _assert_failed(result: subprocess.CompletedProcess):
    assert "simulated low-level failure" in result.stdout + result.stderr, result.stderr
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize("stage", STAGES)
def test_standalone(stage):
    _assert_failed(_run([sys.executable, str(SIMULATE_FATAL_ERROR), stage]))


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is None, reason="mpi4py not installed")
@pytest.mark.parametrize("stage", STAGES)
def test_mpmd(stage):
    """EBFM fails next to a second component that waits for it in a barrier, as in a coupled run."""
    cmd = ["mpirun", "-np", "1", sys.executable, str(SIMULATE_FATAL_ERROR), stage]
    cmd += [":", "-np", "1", sys.executable, "-c", "from mpi4py import MPI; MPI.COMM_WORLD.Barrier()"]
    _assert_failed(_run(cmd))


def test_usage_error_standalone_exits_without_abort():
    result = _run([sys.executable, str(SIMULATE_FATAL_ERROR), "usage-error"])

    assert "simulated usage error" in result.stderr, result.stderr
    assert result.returncode == 2, result.stdout + result.stderr
    assert "MPI_ABORT" not in result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is None, reason="mpi4py not installed")
def test_usage_error_mpmd_aborts():
    """A usage error of EBFM ends the MPMD run instead of leaving the second component waiting in a barrier."""
    cmd = ["mpirun", "-np", "1", sys.executable, str(SIMULATE_FATAL_ERROR), "usage-error"]
    cmd += [":", "-np", "1", sys.executable, "-c", "from mpi4py import MPI; MPI.COMM_WORLD.Barrier()"]
    result = _run(cmd)

    assert "simulated usage error" in result.stderr, result.stderr
    assert result.returncode != 0, result.stdout + result.stderr


def test_no_abort_without_errorcode():
    """mpi4py's Comm.Abort() defaults to errorcode 0, i.e. it reports success."""
    offenders = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Abort"
                and not node.args
                and not node.keywords
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"Abort() without an error code: {offenders}"
