# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the output of a test grid of test_loop_write_to_file with its columns distributed over the MPI ranks.

Usage: mpirun -np N python write_output_on_ranks.py {structured|unstructured} OUTDIR

Rank r holds the columns r, r + N, r + 2N, ... in reverse order, and only the mesh vertices of its own cells, so
that neither the ranks nor their columns and vertices are in id order.
"""

import sys
from pathlib import Path

import numpy as np

from ebfm.core.comm import defaultComm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_loop_write_to_file as fixtures  # noqa: E402

GRIDS = {
    "structured": fixtures._structured_grid,
    "unstructured": fixtures._unstructured_grid_with_mesh,
}

grid_kind, outdir = sys.argv[1], sys.argv[2]
columns = np.arange(fixtures.GPSUM)[defaultComm.rank :: defaultComm.size][::-1]
fixtures._write_single_step(outdir, GRIDS[grid_kind](), columns=columns)
