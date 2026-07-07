# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Comm
else:
    Comm = Any


class _SerialComm:
    rank = 0
    size = 1


try:
    from mpi4py import MPI as _MPI

    mpi_available = True
    defaultComm: Comm = _MPI.COMM_WORLD
except ImportError as e:
    mpi_available = False
    mpi_import_error = e
    defaultComm: Comm = _SerialComm()
