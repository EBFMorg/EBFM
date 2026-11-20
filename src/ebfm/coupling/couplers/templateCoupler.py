# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .base import Coupler, CouplingConfig, CouplerExitCode, GridDict
from ebfm.coupling.fields import FieldSet, GenericExchangeType

import logging

logger = logging.getLogger(__name__)


class TemplateCoupler(Coupler[GenericExchangeType]):
    """
    Template for a new coupler that wraps a coupling library.

    Copy this file, rename the class and implement every method that raises NotImplementedError. Replace the type
    parameter GenericExchangeType with the exchange type of the coupling library (see YACCoupler). Add the coupling
    library to ebfm.core.config.CouplingLibrary, register the new coupler in ebfm.coupling._COUPLERS and select it in
    ebfm.coupling.select_coupler_class().
    """

    def __init__(self, coupling_config: CouplingConfig):
        """
        Initialize the coupling library and register EBFM as a component.

        @param[in] coupling_config coupling configuration of this component
        """
        super().__init__(coupling_config)
        raise NotImplementedError("TemplateCoupler.__init__ is not implemented.")

    @staticmethod
    def get_mpi_handshake_group_name() -> str:
        """
        Get the group name used with this coupler for MPI handshake.

        @returns group name for MPI handshake
        """
        raise NotImplementedError("TemplateCoupler.get_mpi_handshake_group_name is not implemented.")

    def _setup(self, grid: GridDict, field_definitions: FieldSet):
        """
        Register the grid and the coupled fields with the coupling library.

        Must add every coupled field to self._fields, because put() and get() reject fields that are not registered
        there.

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] field_definitions set of field definitions collected from all coupled components
        """
        raise NotImplementedError("TemplateCoupler._setup is not implemented.")

    def _map_exchange_type(self, exchange_type: GenericExchangeType) -> GenericExchangeType:
        """
        Map generic exchange type to the exchange type of the coupling library.
        """
        raise NotImplementedError("TemplateCoupler._map_exchange_type is not implemented.")

    def _put(self, component_name: str, field_name: str, data: np.ndarray) -> CouplerExitCode | None:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns exit code, or None if put successfully completed.
        """
        raise NotImplementedError("TemplateCoupler._put is not implemented.")

    def _get(self, component_name: str, field_name: str) -> tuple[np.ndarray | None, CouplerExitCode | None]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, exit code). Exit code is None if get successfully received data.
        """
        raise NotImplementedError("TemplateCoupler._get is not implemented.")

    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("TemplateCoupler.finalize is not implemented.")
