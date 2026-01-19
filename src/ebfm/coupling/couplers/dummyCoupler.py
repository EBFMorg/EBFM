# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from . import Coupler
from .base import Grid, Dict, CouplingConfig

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

    def setup(self, grid: Grid, time: Dict[str, float]):
        """Setup the coupling interface (does nothing for DummyCoupler)

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        logger.debug("Setup coupling...")
        logger.debug("Do nothing for DummyCoupler.")

    def put(self, component_name: str, field_name: str, data: np.array):
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent
        """
        logger.debug(f"Put field {field_name} to {component_name}...")
        logger.debug("Do nothing for DummyCoupler.")

    def get(self, component_name: str, field_name: str) -> np.array:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns field data
        """
        logger.debug(f"Get field {field_name} from {component_name}...")
        logger.debug("Do nothing for DummyCoupler.")

    def finalize(self):
        """Finalize the coupling interface (does nothing for DummyCoupler)"""
        logger.debug("Finalizing coupling...")
        logger.debug("No coupling to finalize for DummyCoupler.")
