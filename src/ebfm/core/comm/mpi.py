# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from mpi4py import MPI

from ebfm.core.config import CouplingConfig
from ebfm.coupling import select_coupler_class

from .mpi_handshake import mpi_handshake

from ebfm.core.logger import Logger, getLogger

# logger for this module
logger: Logger

logger = getLogger(__name__)


def _is_initialized() -> bool:
    return MPI.Is_initialized()


def abort():
    if _is_initialized():
        logger.info(f"Calling Abort on MPI comm {_global_comm}.")
        _global_comm.Abort()
        logger.info("Aborting successful.")


_global_comm: MPI.Comm | None = None


def _set_global_comm(comm: MPI.Comm):
    global _global_comm
    _global_comm = comm


def do_comm_splitting(
    comp_name: str, coupling_config: CouplingConfig, global_comm: MPI.Comm = MPI.COMM_WORLD
) -> MPI.Comm:
    """
    Perform MPI communicator splitting.

    The MPI communicator splitting will be performed using the mpi-handshake algorithm and it will create a
    communication infrastructure that is compatible with the provided coupling configuration. The returned
    coupler class must be used later to construct the coupling.

    @param[in] comp_name local component name for which communicator is returned.
    @param[in] coupling_config coupling configuration where group communicators are stored.
    @param[in] global_comm global MPI communicator to use for splitting. By default, MPI.COMM_WORLD is used.

    @return local communicator for EBFM and selected coupler class.
    """

    _set_global_comm(global_comm)

    coupler_cls = select_coupler_class(coupling_config)
    coupler_group_name = coupler_cls.get_mpi_handshake_group_name()

    groupnames = set()
    groupnames.add(comp_name)
    groupnames.add(coupler_group_name)
    groupcomms = mpi_handshake(groupnames=list(groupnames), comm=_global_comm)

    coupling_config.set_group_communicators(groupcomms)
    ebfm_comm = groupcomms[comp_name]

    return ebfm_comm, coupler_cls
