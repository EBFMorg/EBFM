# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A restart chain on several MPI ranks gives exactly the results of a restart chain on a single rank.

Each chain runs two days: the first writes a restart file, the second starts from it. The restart files and the
results of the second day must not depend on the number of ranks of either leg.
"""

import filecmp
from pathlib import Path

import pytest
from test_init import copy_without_column_ids
from test_parallel_runs import CASES, _assert_same_results, _run, requires_mpirun

_LEG_1 = ["--start-time", "1979-07-01T00:00:00Z", "--end-time", "1979-07-02T00:00:00Z"]
_LEG_2 = ["--start-time", "1979-07-02T00:00:00Z", "--end-time", "1979-07-03T00:00:00Z", "--restart-init"]
RESTART_FILES = ("restart_1979-07-02T00:00:00Z.nc", "restart_1979-07-03T00:00:00Z.nc")


def _leg(case: str, leg: list[str], n_ranks: int, chain: Path) -> Path:
    """Run one leg of a restart chain in chain, with the chain's restart directory; return the leg's workdir."""
    workdir = chain / ("leg1" if leg is _LEG_1 else "leg2")
    return _run([*CASES[case], *leg, "--restart-dir", str(chain / "restart")], n_ranks, workdir)


@requires_mpirun
@pytest.mark.parametrize("case, n_ranks_per_leg", [("matlab", (4, 3)), ("elmer", (3, 2))], ids=str)
def test_parallel_restart_chain_equals_serial_chain(tmp_path, case, n_ranks_per_leg):
    for leg in (_LEG_1, _LEG_2):
        _leg(case, leg, 1, tmp_path / "serial")
    for leg, n_ranks in zip((_LEG_1, _LEG_2), n_ranks_per_leg):
        _leg(case, leg, n_ranks, tmp_path / "parallel")

    for restart_file in RESTART_FILES:
        serial_file = tmp_path / "serial" / "restart" / restart_file
        assert filecmp.cmp(serial_file, tmp_path / "parallel" / "restart" / restart_file, shallow=False), restart_file
    _assert_same_results(tmp_path / "serial" / "leg2", tmp_path / "parallel" / "leg2")


@requires_mpirun
def test_restart_file_without_column_ids_on_several_ranks(tmp_path):
    for leg in (_LEG_1, _LEG_2):
        _leg("matlab", leg, 1, tmp_path / "serial")
    (tmp_path / "parallel" / "restart").mkdir(parents=True)
    copy_without_column_ids(
        tmp_path / "serial" / "restart" / RESTART_FILES[0], tmp_path / "parallel" / "restart" / RESTART_FILES[0]
    )

    parallel = _leg("matlab", _LEG_2, 3, tmp_path / "parallel")

    _assert_same_results(tmp_path / "serial" / "leg2", parallel)
