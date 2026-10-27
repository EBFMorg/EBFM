# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check domain_mean on 3 MPI ranks that hold the 10 columns of a domain unevenly, rank 2 holding none.

Usage: mpirun -np 3 python domain_mean.py
"""

import numpy as np

from ebfm.core.comm import defaultComm as comm
from ebfm.core.grid import domain_mean

assert comm.size == 3, f"Run on 3 ranks, not {comm.size}."

values = 60.0 + 2.5 * np.arange(10)
columns = {0: [5, 1, 7, 3], 1: [0, 9, 2, 8, 4, 6], 2: []}[comm.rank]

mean = domain_mean(values[columns], comm)

np.testing.assert_allclose(mean, np.mean(values), rtol=1e-15)
assert comm.allgather(mean) == [mean] * comm.size, "Ranks got different means."
print(f"rank {comm.rank}: ok")
