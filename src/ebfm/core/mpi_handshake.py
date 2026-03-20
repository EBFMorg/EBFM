# Copyright (c) 2025 The mpi-handshake Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# From https://gitlab.dkrz.de/dkrz-sw/mpi-handshake

from mpi4py import MPI
import numpy as np


def mpi_handshake(groupnames: list[str], comm: MPI.Comm = MPI.COMM_WORLD) -> dict[str, MPI.Comm]:
    comms = {}
    version = np.array(1, dtype=np.int32)
    comm.Allreduce(MPI.IN_PLACE, (version, 1, MPI.INT), op=MPI.MIN)
    if version != 1:
        MPI.Abort()
    while True:
        broadcaster = np.array(comm.size, dtype=np.int32)
        if len(groupnames) > 0:
            broadcaster[()] = comm.rank
        comm.Allreduce(MPI.IN_PLACE, (broadcaster, 1, MPI.INT), op=MPI.MIN)
        if broadcaster >= comm.size:
            break

        glen = np.array(0, np.int32)
        if broadcaster == comm.rank:
            glen[()] = len(groupnames[0])
        comm.Bcast((glen, 1, MPI.INT), root=broadcaster)

        gnamebuf = np.empty(glen, dtype=np.byte)
        if broadcaster == comm.rank:
            gnamebuf[:] = list(groupnames[0].encode())

        comm.Bcast((gnamebuf, glen, MPI.CHAR), root=broadcaster)
        gname = gnamebuf.tobytes().decode()

        if gname in groupnames:
            gcomm = comm.Split(color=0)
            comms[gname] = gcomm
            groupnames.remove(gname)
        else:
            comm.Split(color=MPI.UNDEFINED)
    return comms
