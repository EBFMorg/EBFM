# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
import numpy as np

from elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?
from ebfm.config import CouplingConfig

import logging

from abc import ABC, abstractmethod

from coupling.helpers import check_coupling_support
from coupling.components import Component

# TODO: should not be necessary if ElmerIce etc. use a generic Field instead of (YAC)Field
coupling_supported: bool
try:
    check_coupling_support()
    coupling_supported = True
except ImportError:
    coupling_supported = False

if coupling_supported:
    from coupling.components import ElmerIce, IconAtmo

logger = logging.getLogger(__name__)


class Coupler(ABC):
    """
    Abstract base class for couplers. Implements the strategy pattern to support different coupling libraries.
    """

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
        self.component_name: str = coupling_config.component_name  # name of this component in the coupler configuration
        self._coupled_components: Dict[str, Component] = {}

        if coupling_supported:
            if coupling_config.couple_to_elmer_ice:
                elmer_comp = ElmerIce(self)
                self._coupled_components[elmer_comp.name] = elmer_comp

            if coupling_config.couple_to_icon_atmo:
                icon_comp = IconAtmo(self)
                self._coupled_components[icon_comp.name] = icon_comp

    def has_coupling_to(self, component_name: str) -> bool:
        """
        Check if coupling to a specific component is enabled

        @param[in] component_name name of the component to check coupling for

        @returns True if coupling to the specified component is enabled, False otherwise
        """
        return component_name in self._coupled_components

    @abstractmethod
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

    @abstractmethod
    def put(self, component_name: str, field_name: str, data: np.array):
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be exchanged
        """
        raise NotImplementedError("put method must be implemented in subclasses.")

    @abstractmethod
    def get(self, component_name: str, field_name: str) -> np.array:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns field data
        """
        raise NotImplementedError("get method must be implemented in subclasses.")

    @abstractmethod
    def exchange(self, component_name: str, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with another component

        @param[in] component_name name of the component to exchange data with
        @param[in] data_to_exchange dictionary of field names and their data to be exchanged

        @returns dictionary of exchanged field data
        """
        raise NotImplementedError("Generic exchange method is not implemented. Use specific exchange methods.")

    @abstractmethod
    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("finalize method must be implemented in subclasses.")
