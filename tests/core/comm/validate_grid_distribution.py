# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check validate_grid_distribution on 3 MPI ranks that hold the 10 columns of a domain unevenly.

Usage: mpirun -np 3 python validate_grid_distribution.py {distinct|duplicate}

distinct: every column is held by one rank, and rank 2 holds none. Checks that no rank raises.
duplicate: rank 2 holds a column that rank 0 holds as well. Checks that every rank raises a ValueError.
"""

import sys

import numpy as np

from ebfm.core.comm import defaultComm as comm
from ebfm.core.grid import validate_grid_distribution

assert comm.size == 3, f"Run on 3 ranks, not {comm.size}."
mode = sys.argv[1]

column_ids = 100 + 7 * np.arange(10)
columns = {0: [5, 1, 7, 3], 1: [0, 9, 2, 8, 4, 6], 2: []}[comm.rank]
if mode == "duplicate" and comm.rank == 2:
    columns = [1]
grid = {"column_id": column_ids[columns]}

try:
    validate_grid_distribution(grid, comm)
except ValueError as e:
    print(f"rank {comm.rank}: {e}")
    sys.exit(0)

print(f"rank {comm.rank}: ok")
