# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""A restart chain on several MPI ranks gives exactly the results of a restart chain on a single rank.

Each chain runs two days: the first writes a restart file, the second starts from it. A run on several ranks writes
one restart file per partition; merged with ebfm.io.merge, they equal the restart file of a run on a single rank, and
split with ebfm.io.split_restart for another partitioning, the second day continues on another number of ranks.
Restart files of a partitioned Elmer mesh are also tested in test_partitioned_mesh_run.
"""

from pathlib import Path

from test_parallel_runs import (
    CASES,
    _assert_same_results,
    _run,
    assert_same_variables,
    parallel_args,
    requires_mpirun,
    tool,
)

from ebfm.grid.structured.partitioning import partition_grid

_LEG_1 = ["--start-time", "1979-07-01T00:00:00Z", "--end-time", "1979-07-02T00:00:00Z"]
_LEG_2 = ["--start-time", "1979-07-02T00:00:00Z", "--end-time", "1979-07-03T00:00:00Z", "--restart-init"]
RESTART_FILES = ("restart_1979-07-02T00:00:00Z.nc", "restart_1979-07-03T00:00:00Z.nc")


def _leg(args: list[str], leg: list[str], n_ranks: int, chain: Path) -> Path:
    """Run one leg of a restart chain in chain, with the chain's restart directory; return the leg's workdir."""
    workdir = chain / ("leg1" if leg is _LEG_1 else "leg2")
    return _run([*args, *leg, "--restart-dir", str(chain / "restart")], n_ranks, workdir)


def _serial_chain(tmp_path: Path) -> Path:
    for leg in (_LEG_1, _LEG_2):
        _leg(CASES["matlab"], leg, 1, tmp_path / "serial")
    return tmp_path / "serial"


@requires_mpirun
def test_restart_chain_on_4_then_3_ranks_equals_serial_chain(tmp_path):
    serial = _serial_chain(tmp_path)
    args = parallel_args("matlab", tmp_path)

    _leg(args, _LEG_1, 4, tmp_path / "np4")
    restart_1 = tmp_path / "np4" / "restart" / Path(RESTART_FILES[0]).stem / "partitioning.4"
    tool("merge", restart_1, "-o", tmp_path / "merged_1.nc")
    assert_same_variables(serial / "restart" / RESTART_FILES[0], tmp_path / "merged_1.nc")

    grid_file = Path(args[args.index("--matlab-mesh") + 1])
    partitioning = partition_grid(grid_file, 3)
    tool(
        "split_restart",
        tmp_path / "merged_1.nc",
        partitioning,
        "-o",
        tmp_path / "np3" / "restart" / restart_1.parent.name / "partitioning.3",
    )
    _leg(args, _LEG_2, 3, tmp_path / "np3")

    restart_2 = tmp_path / "np3" / "restart" / Path(RESTART_FILES[1]).stem / "partitioning.3"
    tool("merge", restart_2, "-o", tmp_path / "merged_2.nc")
    assert_same_variables(serial / "restart" / RESTART_FILES[1], tmp_path / "merged_2.nc")
    _assert_same_results(serial / "leg2", tmp_path / "np3" / "leg2")
