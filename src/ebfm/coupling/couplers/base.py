# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Optional, Tuple
import numpy as np
from enum import Enum

from ebfm.elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.core.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?
from ebfm.core.config import CouplingConfig

import logging

from abc import ABC, abstractmethod

from ebfm.coupling.components import Component

# TODO: should not be necessary if ElmerIce etc. use a generic Field instead of (YAC)Field
from .helpers import coupling_supported

if coupling_supported:
    from ebfm.coupling.components import ElmerIce, IconAtmo

logger = logging.getLogger(__name__)


class CouplerErrorCode(Enum):
    """
    Error codes returned by Coupler.get() and Coupler.put().

    A value of None (i.e. no error code) indicates success.
    """

    WRONG_EXCHANGE_TYPE = "wrong_exchange_type"
    """The field's declared exchange type does not match the operation (SOURCE vs TARGET)."""

    WRONG_ROLE = "wrong_role"
    """The field's actual role in the coupler config does not match its declared role."""


class Coupler(ABC):
    """
    Abstract base class for couplers. Implements the strategy pattern to support different coupling libraries.
    """

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
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

    def get_component(self, component_name: str) -> Component:
        """
        Get the component object for a specific coupled component

        @param[in] component_name name of the component to get

        @returns Component object for the specified component

        @raises KeyError if the specified component is not coupled
        """
        if not self.has_coupling_to(component_name):
            raise KeyError(f"Coupling to component '{component_name}' is not enabled.")
        return self._coupled_components[component_name]

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
    def put(self, component_name: str, field_name: str, data: np.array) -> Optional[CouplerErrorCode]:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns error code, or None if no error occurred.
        """
        raise NotImplementedError("put method must be implemented in subclasses.")

    @abstractmethod
    def get(self, component_name: str, field_name: str) -> Tuple[Optional[np.array], Optional[CouplerErrorCode]]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, error code). Error code is None if no error occurred.
        """
        raise NotImplementedError("get method must be implemented in subclasses.")

    @abstractmethod
    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("finalize method must be implemented in subclasses.")
