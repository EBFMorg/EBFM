# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar, Generic
import numpy as np
from enum import Enum

from ebfm.core.grid import GridDict
from ebfm.core.constants import DAYS_PER_YEAR

from ebfm.elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.core.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?
from ebfm.core.config import CouplingConfig

import logging

from abc import ABC, abstractmethod

from ebfm.coupling.components import Component, ElmerIce, IconAtmo
from ebfm.coupling.fields import FieldSet, GenericExchangeType

logger = logging.getLogger(__name__)

CouplerExchangeType = TypeVar("CouplerExchangeType")
"""Backend-specific exchange type used by a concrete coupler implementation.

Examples:
- Generic/dummy couplers may use the generic `GenericExchangeType` directly.
- YAC coupler uses `yac.ExchangeType`.
"""


class CouplerErrorCode(Enum):
    """
    Error codes returned by Coupler.get() and Coupler.put().

    A value of None (i.e. no error code) indicates success.
    """

    WRONG_EXCHANGE_TYPE = "wrong_exchange_type"
    """The field's declared exchange type does not match the operation (SOURCE vs TARGET)."""

    WRONG_ROLE = "wrong_role"
    """The field's actual role in the coupler config does not match its declared role."""


class Coupler(ABC, Generic[CouplerExchangeType]):
    """
    Abstract base class for couplers. Implements the strategy pattern to support different coupling libraries.

    Why this class is generic:
    - Components and field definitions use backend-independent exchange roles (`GenericExchangeType`).
    - Concrete couplers can require backend-specific role types (e.g. `yac.ExchangeType`).
    - `_map_exchange_type` bridges generic roles to backend-specific roles.

    This keeps component code backend-agnostic while allowing strict backend checks in coupler implementations.
    """

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
        self._coupled_components: dict[str, Component] = {}

        if coupling_config.couple_to_elmer_ice:
            elmer_comp = ElmerIce(self)
            self._coupled_components[elmer_comp.name] = elmer_comp

        if coupling_config.couple_to_icon_atmo:
            icon_comp = IconAtmo(self)
            self._coupled_components[icon_comp.name] = icon_comp

        self.fields: FieldSet = FieldSet()
        self._time: dict[str, float] | None = None

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

    def setup(self, grid: GridDict, time: dict[str, float]):
        """
        Setup the coupling interface.

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        self._time = time

        field_definitions = FieldSet()

        for component in self._coupled_components.values():
            field_definitions |= component.get_field_definitions(self._time)

        self._setup(grid, field_definitions)

    @abstractmethod
    def _setup(self, grid: GridDict, field_definitions: FieldSet):
        """
        Perform coupler-specific setup tasks such as:
        - Adding the grid to the coupler interface
        - Adding coupled fields to the coupler interface based on component definitions

        @param[in] grid grid information dictionary
        @param[in] field_definitions set of field definitions collected from all coupled components
        """
        raise NotImplementedError("_setup method must be implemented in subclasses.")

    def get_conversion_per_year_factor(self) -> float:
        """
        Returns the conversion factor to scale a per-timestep value to a per-year value.

        Performs the calculation based on the current time step size stored in the instance.

        @note This method assumes that self.time has been set and contains the key 'dt'.

        @return The conversion factor (unitless).
        """
        assert self._time is not None, "Time information must be set before calling get_conversion_per_year_factor."
        assert "dt" in self._time, "Time information must include 'dt' key representing the time step size."

        ebfm_time_step = self._time["dt"]  # time step size in days

        return DAYS_PER_YEAR / ebfm_time_step

    def _add_grid(self, grid_name: str, grid: Grid):
        """
        Add grid to the Coupler interface
        """
        raise NotImplementedError("add_grid method must be implemented in subclasses.")

    def _add_couples(self, field_definitions: FieldSet):
        """
        Add coupling definitions to the Coupler interface
        """
        raise NotImplementedError("add_couples method must be implemented in subclasses.")

    @abstractmethod
    def _map_exchange_type(self, exchange_type: GenericExchangeType) -> CouplerExchangeType:
        """
        Map generic exchange type to backend-specific exchange type representation.
        """
        raise NotImplementedError("_map_exchange_type must be implemented in subclasses.")

    @abstractmethod
    def put(self, component_name: str, field_name: str, data: np.ndarray) -> CouplerErrorCode | None:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns error code, or None if no error occurred.
        """
        raise NotImplementedError("put method must be implemented in subclasses.")

    @abstractmethod
    def get(self, component_name: str, field_name: str) -> tuple[np.ndarray | None, CouplerErrorCode | None]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, error code). Error code is None if no error occurred.
        """
        raise NotImplementedError("get method must be implemented in subclasses.")

    def has_field(self, component_name: str, field_name: str, exchange_type: GenericExchangeType) -> bool:
        """
        Check whether a field with given name and exchange type exists for a coupled component.

        @param[in] component_name name of the component
        @param[in] field_name name of the field
        @param[in] exchange_type expected exchange type

        @returns True if such a field exists, otherwise False
        """
        if not self.has_coupling_to(component_name):
            return False

        expected_exchange_type = self._map_exchange_type(exchange_type)
        component = self._coupled_components[component_name]
        fields = self.fields.filter(
            lambda f: f.coupled_component == component
            and f.name == field_name
            and f.exchange_type == expected_exchange_type
        )
        return not fields.is_empty()

    @abstractmethod
    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("finalize method must be implemented in subclasses.")
