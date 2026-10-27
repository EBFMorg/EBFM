# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check ColumnGather on 3 MPI ranks that hold the 10 columns of a domain unevenly and out of id order.

Usage: mpirun -np 3 python gather_columns.py

Every column is held by one rank, and rank 2 holds none. Checks the arrays gathered on rank 0.
"""

import numpy as np

from ebfm.core.comm import defaultComm as comm
from ebfm.core.comm.gather import ColumnGather

assert comm.size == 3, f"Run on 3 ranks, not {comm.size}."

column_ids = 100 + 7 * np.arange(10)
columns = {0: [5, 1, 7, 3], 1: [0, 9, 2, 8, 4, 6], 2: []}[comm.rank]
local_ids = column_ids[columns]

gather = ColumnGather(comm, local_ids)
values = gather.gather(1.5 * local_ids)
layers = gather.gather(np.stack([local_ids, 2 * local_ids], axis=1))
concatenated = gather.gather_unordered(local_ids)

if gather.is_root:
    np.testing.assert_array_equal(gather.column_ids, column_ids)
    np.testing.assert_array_equal(values, 1.5 * column_ids)
    np.testing.assert_array_equal(layers, np.stack([column_ids, 2 * column_ids], axis=1))
    np.testing.assert_array_equal(concatenated, column_ids[[5, 1, 7, 3, 0, 9, 2, 8, 4, 6]])
else:
    assert gather.column_ids is None and values is None and layers is None and concatenated is None

print(f"rank {comm.rank}: ok")
