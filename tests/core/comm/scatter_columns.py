# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check ColumnGather's scatters on 3 MPI ranks that hold the 10 columns of a domain unevenly and out of id order.

Usage: mpirun -np 3 python scatter_columns.py

Every column is held by one rank, and rank 2 holds none. Checks the rows every rank receives from rank 0.
"""

import numpy as np

from ebfm.core.comm import defaultComm as comm
from ebfm.core.comm.gather import ColumnGather

assert comm.size == 3, f"Run on 3 ranks, not {comm.size}."

column_ids = 100 + 7 * np.arange(10)
columns = {0: [5, 1, 7, 3], 1: [0, 9, 2, 8, 4, 6], 2: []}[comm.rank]
local_ids = column_ids[columns]

gather = ColumnGather(comm, local_ids)
is_root = gather.is_root
values = gather.scatter(1.5 * column_ids if is_root else None)
layers = gather.scatter(np.stack([column_ids, 2 * column_ids], axis=1) if is_root else None)
blocks = gather.scatter_unordered(np.arange(10) if is_root else None)
round_trip = gather.scatter(gather.gather(-local_ids))

np.testing.assert_array_equal(values, 1.5 * local_ids)
np.testing.assert_array_equal(layers, np.stack([local_ids, 2 * local_ids], axis=1).reshape(-1, 2))
np.testing.assert_array_equal(blocks, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9], 2: []}[comm.rank])
np.testing.assert_array_equal(round_trip, -local_ids)

print(f"rank {comm.rank}: ok")
