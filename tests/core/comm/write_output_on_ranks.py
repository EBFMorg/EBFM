# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the output of the structured test grid of test_loop_write_to_file with its columns distributed over the MPI
ranks.

Usage: mpirun -np N python write_output_on_ranks.py OUTDIR

Rank r holds the columns r, r + N, r + 2N, ... in reverse order, so that neither the ranks nor their columns are in id
order.
"""

import sys
from pathlib import Path

import numpy as np

from ebfm.core.comm import defaultComm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_loop_write_to_file as fixtures  # noqa: E402

outdir = sys.argv[1]
columns = np.arange(fixtures.GPSUM)[defaultComm.rank :: defaultComm.size][::-1]
fixtures._write_single_step(outdir, fixtures._structured_grid(), columns=columns)
