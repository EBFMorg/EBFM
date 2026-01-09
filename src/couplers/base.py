# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
import numpy as np

from elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?
from ebfm.config import CouplingConfig

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Component(Enum):
    """
    Other components which this model can couple to
    """

    elmer_ice = auto()
    icon_atmo = auto()


class Coupler:
    """
    Abstract base class for couplers. Implements the strategy pattern to support different coupling libraries.
    """

    component_name: str  # name of this component in the coupler configuration

    _couples_to: Dict[Component, bool] = {}  # dict gives information whether to couple with given component

    def has_coupling_to(self, component_name: str) -> bool:
        """
        Check if coupling to a specific component is enabled

        @param[in] component_name name of the component to check coupling for

        @returns True if coupling to the specified component is enabled, False otherwise
        """
        component = Component[component_name]
        return self._couples_to[component]

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
        raise NotImplementedError("Coupler is an abstract base class and cannot be instantiated directly.")

    def setup(self, grid: Grid, time: Dict[str, float]):
        raise NotImplementedError("setup method must be implemented in subclasses.")

    def _add_grid(self, grid_name: str, grid: Grid):
        """
        Add grid to the Coupler interface
        """
        raise NotImplementedError("add_grid method must be implemented in subclasses.")

    def _add_couples(self, time: Dict[str, float]):
        """
        Add coupling definitions to the Coupler interface
        """
        raise NotImplementedError("add_couples method must be implemented in subclasses.")

    def exchange(self, component_name: str, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with another component

        @param[in] component_name name of the component to exchange data with
        @param[in] data_to_exchange dictionary of field names and their data to be exchanged

        @returns dictionary of exchanged field data
        """
        raise NotImplementedError("Generic exchange method is not implemented. Use specific exchange methods.")

    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("finalize method must be implemented in subclasses.")
