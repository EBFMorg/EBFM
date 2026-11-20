# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .base import Coupler, CouplingConfig, CouplerExitCode, GridDict
from ebfm.coupling.fields import FieldSet, GenericExchangeType

import logging

logger = logging.getLogger(__name__)


class OASISCoupler(Coupler[GenericExchangeType]):
    """
    Coupler implementation for the OASIS coupling library.

    Not implemented yet: this is TemplateCoupler renamed for OASIS. See TemplateCoupler for what each method has to do.
    """

    def __init__(self, coupling_config: CouplingConfig):
        """
        Initialize OASIS and register EBFM as a component.

        @param[in] coupling_config coupling configuration of this component
        """
        super().__init__(coupling_config)
        raise NotImplementedError("OASISCoupler.__init__ is not implemented.")

    @staticmethod
    def get_mpi_handshake_group_name() -> str:
        """
        Get the group name used with this coupler for MPI handshake.

        @returns group name for MPI handshake
        """
        raise NotImplementedError("OASISCoupler.get_mpi_handshake_group_name is not implemented.")

    def _setup(self, grid: GridDict, field_definitions: FieldSet):
        """
        Register the grid and the coupled fields with OASIS.

        Must add every coupled field to self._fields, because put() and get() reject fields that are not registered
        there.

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] field_definitions set of field definitions collected from all coupled components
        """
        raise NotImplementedError("OASISCoupler._setup is not implemented.")

    def _map_exchange_type(self, exchange_type: GenericExchangeType) -> GenericExchangeType:
        """
        Map generic exchange type to the exchange type of OASIS.
        """
        raise NotImplementedError("OASISCoupler._map_exchange_type is not implemented.")

    def _put(self, component_name: str, field_name: str, data: np.ndarray) -> CouplerExitCode | None:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns exit code, or None if put successfully completed.
        """
        raise NotImplementedError("OASISCoupler._put is not implemented.")

    def _get(self, component_name: str, field_name: str) -> tuple[np.ndarray | None, CouplerExitCode | None]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, exit code). Exit code is None if get successfully received data.
        """
        raise NotImplementedError("OASISCoupler._get is not implemented.")

    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("OASISCoupler.finalize is not implemented.")
