# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check --dump-reference on 3 MPI ranks that hold the 10 columns of a domain unevenly and out of id order.

Usage: mpirun -np 3 python dump_reference_on_ranks.py FILE

Rank 0 must write the arrays of all columns, in column id order, to FILE.
"""

import logging
import sys

import numpy as np

from ebfm.core.comm import defaultComm as comm
from ebfm.core.comm.gather import ColumnGather
from ebfm.main import dump_reference

assert comm.size == 3, f"Run on 3 ranks, not {comm.size}."
filepath = sys.argv[1]

column_ids = 100 + 7 * np.arange(10)
columns = {0: [5, 1, 7, 3], 1: [0, 9, 2, 8, 4, 6], 2: []}[comm.rank]
local_ids = column_ids[columns]
OUT = {"smb": 0.5 * local_ids, "subT": np.stack([local_ids, 2 * local_ids], axis=1).astype(np.float64)}

dump_reference(logging.getLogger(__name__), OUT, filepath, ColumnGather(comm, local_ids))

if comm.rank == 0:
    with np.load(filepath) as data:
        assert sorted(data.files) == ["smb", "subT"], data.files
        np.testing.assert_array_equal(data["smb"], 0.5 * column_ids)
        np.testing.assert_array_equal(data["subT"], np.stack([column_ids, 2 * column_ids], axis=1))
print(f"rank {comm.rank}: ok")
