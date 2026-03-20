# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dummy MPI handshake participant used for testing.

Joins two MPI groups (dummy and coupler) via mpi handshake, checks ranks and sizes, and asserts
correct group membership and ordering based on invocation order.
"""

from ebfm.core.mpi_handshake import mpi_handshake
import sys

# local communicator for dummy this executable
DUMMY_LABEL = "mpi_handshake_dummy"

# join the "ebfmDummyCoupler" group for testing purposes
CPL_LABEL = "ebfmDummyCoupler"

comms = mpi_handshake([DUMMY_LABEL, CPL_LABEL])

dummy_comm = comms[DUMMY_LABEL]

print(f"mpi_handshake_dummy.py: Hello from rank {dummy_comm.rank} in group {DUMMY_LABEL} with size {dummy_comm.size}!")
assert dummy_comm.size == 1, "This dummy test should be run with exactly one process in the dummy group."
assert dummy_comm.rank == 0, "This dummy test should be run with exactly one process in the dummy group."

assert len(sys.argv) == 2, "This test should be run as 'python .../mpi_handshake_dummy.py <first|second>'."
if sys.argv[1] == "first":
    expected_rank = 0  # since dummy is launched before EBFM, it should have rank 0 in the CPL group
elif sys.argv[1] == "second":
    expected_rank = 1  # since dummy is launched after EBFM, it should have rank 1 in the CPL group
else:
    raise ValueError(f"Expected argument 'first' or 'second' to determine expected rank in group {CPL_LABEL}.")

cpl_comm = comms[CPL_LABEL]
print(f"mpi_handshake_dummy.py: Hello from rank {cpl_comm.rank} in group {CPL_LABEL} with size {cpl_comm.size}!")
assert cpl_comm.size == 2, f"Group {CPL_LABEL} should have the EBFM procs plus the procs from the dummy group."
assert cpl_comm.rank == expected_rank, f"This executable is launched {sys.argv[1]}, should have rank {expected_rank}."
