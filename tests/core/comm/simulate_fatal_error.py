# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run ebfm.main.main() with a _main_impl() that fails like a low-level library error.

Usage: python simulate_fatal_error.py [after-comm-splitting|before-comm-splitting|usage-error]

after-comm-splitting (default): fails once EBFM's global MPI communicator is set, like an error in the
time loop. before-comm-splitting: fails before that, like an error while reading the configuration.
usage-error: exits with status 2 once the communicator is set, like a rejected command line option.
The script runs standalone or as one component of an MPMD run under mpirun.
"""

import sys

import ebfm.core.comm
import ebfm.main

STAGES = ("after-comm-splitting", "before-comm-splitting", "usage-error")

stage = sys.argv[1] if len(sys.argv) > 1 else STAGES[0]
assert stage in STAGES, f"Expected one of {STAGES}, got '{stage}'."


def _failing_main_impl():
    if stage != "before-comm-splitting" and ebfm.core.comm.mpi_available:
        from mpi4py import MPI

        from ebfm.core.comm import mpi as comm_mpi

        comm_mpi._set_global_comm(MPI.COMM_WORLD)

    if stage == "usage-error":
        print("simulated usage error", file=sys.stderr)
        sys.exit(2)
    raise ValueError("simulated low-level failure")


ebfm.main._main_impl = _failing_main_impl
ebfm.main.main()
