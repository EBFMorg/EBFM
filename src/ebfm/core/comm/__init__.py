# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from ebfm.core.logger import Logger
from ebfm.core.config import CouplingConfig
from ebfm.coupling import Coupler, select_coupler_class

from .mpi_handshake import mpi_handshake

from mpi4py import MPI

# logger for this module
logger: Logger


def do_comm_splitting(comp_name: str, coupling_config: CouplingConfig) -> tuple[MPI.Comm, type[Coupler]]:
    """
    Perform MPI communicator splitting.

    The MPI communicator splitting will be performed using the mpi-handshake algorithm and it will create a
    communication infrastructure that is compatible with the provided coupling configuration. The returned
    coupler class must be used later to construct the coupling.

    @param[in] comp_name local component name for which communicator is returned.
    @param[in] coupling_config coupling configuration where group communicators are stored.

    @return local communicator for EBFM and selected coupler class.
    """
    coupler_cls = select_coupler_class(coupling_config)

    groupnames = set()
    groupnames.add(comp_name)
    groupnames.add(coupler_cls.get_mpi_handshake_group_name())
    groupcomms = mpi_handshake(groupnames=list(groupnames), comm=MPI.COMM_WORLD)

    coupling_config.set_group_communicators(groupcomms)
    ebfm_comm = groupcomms[comp_name]

    return ebfm_comm, coupler_cls
