# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from . import Coupler
from .base import GridDict, CouplingConfig, CouplerErrorCode
from ebfm.coupling.fields import GenericExchangeType, FieldSet

import logging

logger = logging.getLogger(__name__)


class DummyCoupler(Coupler):
    """
    A dummy coupler implementation that does nothing.
    This can be used when no coupling is required.
    """

    def __init__(self, coupling_config: CouplingConfig):
        super().__init__(coupling_config)

        # DummyCoupler couples to none of the available components
        self._coupled_components = dict()
        logger.debug(f"DummyCoupler created for component '{coupling_config.component_name}'.")

    def _setup(self, grid: GridDict, field_definitions: FieldSet):
        """DummyCoupler has no specific setup operations.

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] field_definitions set of field definitions collected from all coupled components
        """
        logger.debug("Setup coupling...")
        logger.debug("Do nothing for DummyCoupler.")

    def _map_exchange_type(self, exchange_type: GenericExchangeType) -> GenericExchangeType:
        """
        Dummy coupler keeps generic exchange types unchanged.
        """
        return exchange_type

    def put(self, component_name: str, field_name: str, data: np.ndarray) -> CouplerErrorCode | None:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns error code, or None if no error occurred.
        """
        logger.debug(f"Put field {field_name} to {component_name}...")
        logger.debug("Do nothing for DummyCoupler.")
        return None

    def get(self, component_name: str, field_name: str) -> tuple[np.ndarray | None, CouplerErrorCode | None]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, error code). Error code is None if no error occurred.
        """
        logger.debug(f"Get field {field_name} from {component_name}...")
        logger.debug("Do nothing for DummyCoupler.")
        return None, None

    def finalize(self):
        """Finalize the coupling interface (does nothing for DummyCoupler)"""
        logger.debug("Finalizing coupling...")
        logger.debug("No coupling to finalize for DummyCoupler.")
