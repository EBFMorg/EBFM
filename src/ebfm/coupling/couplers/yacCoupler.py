# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import yac
import numpy as np

from ebfm.core import logging

from .base import Coupler, Grid, Dict, CouplingConfig, CouplerErrorCode

# from coupling import Field  # TODO: rather use generic Field from coupling
from ebfm.coupling.fields import FieldSet
from ebfm.coupling.fields import YACField

from typing import Tuple, Optional

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?

logger = logging.getLogger(__name__)


class YACCoupler(Coupler):
    def __init__(self, coupling_config: CouplingConfig):
        """
        Create interface to the coupler and register component

        @param[in] coupling_config coupling configuration of this component
        """
        super().__init__(coupling_config)
        logger.debug(f"YAC version is {yac.version()}")
        self.interface: yac.YAC = yac.YAC()

        if coupling_config.coupler_config:
            self.interface.read_config_yaml(str(coupling_config.coupler_config))

        self.component: yac.Component = self.interface.def_comp(coupling_config.component_name)
        self.fields: FieldSet = FieldSet()
        self.field_validation_level = coupling_config.field_validation_level

        # will be initialized in self._add_grid()
        self.grid: yac.UnstructuredGrid = None
        self.corner_points: yac.Points = None

    def setup(self, grid: Dict | Grid, time: Dict[str, float]):
        """
        Setup the coupling interface

        Performs initialization operations after init and before entering the
        time loop

        @note This is a collective operation for all components involved in the coupling. It may take some time as it
              involves significant communication and computes remapping weights.

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        grid_name = "ebfm_grid"  # TODO: get from ebfm_coupling_config?

        self._add_grid(grid_name, grid)

        field_definitions = FieldSet()

        for component in self._coupled_components.values():
            field_definitions |= component.get_field_definitions(time)

        self._add_couples(field_definitions)

        self.interface.enddef()

        for field in self.fields.all():
            assert isinstance(field, YACField), f"Expected YACField, got {type(field)}"
            logger.debug(f"Performing consistency checks for field '{field.name}'...")
            field.perform_consistency_checks(self.interface, self.field_validation_level)

    def _get_field(self, component_name: str, field_name: str) -> YACField:
        """
        Get YACField object for given component and field name

        @param[in] component_name name of the component
        @param[in] field_name name of the field

        @returns YACField object
        """

        assert self.has_coupling_to(
            component_name
        ), f"Cannot get field for {component_name} because no coupling exists."

        component = self._coupled_components[component_name]

        comp_fields = self.fields.filter(lambda f: f.coupled_component == component and f.name == field_name).all()

        if len(comp_fields) == 0:
            raise KeyError(f"No field named '{field_name}' found for component '{component_name}'.")
        elif len(comp_fields) > 1:
            raise KeyError(
                f"Found {len(comp_fields)} fields named '{field_name}' found for component '{component_name}'. "
                f"Expected exactly one field per component and field name."
            )

        field = comp_fields.pop()
        assert isinstance(field, YACField), f"Expected YACField, got {type(field)}"
        return field

    def put(self, component_name: str, field_name: str, data: np.ndarray) -> Optional[CouplerErrorCode]:
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent

        @returns error code, or None if no error occurred.
        """

        field = self._get_field(component_name, field_name)

        # Check field exchange type and handle according to validation level
        if field.exchange_type != yac.ExchangeType.SOURCE:
            error_msg = (
                f"Cannot put data for field '{field.name}' of component '{field.coupled_component.name}'. "
                f"Field has to be a SOURCE field, but its exchange_type={field.exchange_type}."
            )
            self._handle_field_validation_error(error_msg)
            return CouplerErrorCode.WRONG_EXCHANGE_TYPE  # If we didn't raise, skip the put operation

        logger.debug(f"Sending field {field.name} to {field.coupled_component.name}...")
        assert field.yac_field is not None, f"YAC field for '{field.name}' has not been created yet."
        field.yac_field.put(data)
        logger.debug(f"Sending field {field.name} to {field.coupled_component.name} complete.")
        return None

    def get(self, component_name: str, field_name: str) -> Tuple[Optional[np.ndarray], Optional[CouplerErrorCode]]:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns tuple of (field data, error code). Error code is None if no error occurred.
                 Field data is always the value returned by YAC (e.g. zeros as fallback) even
                 when an error code is set, so the caller can decide whether to use it or substitute
                 their own fallback.
        """

        field = self._get_field(component_name, field_name)
        error = None
        expected_role = yac.ExchangeType.TARGET

        # Check field exchange type and handle according to validation level
        if field.exchange_type != expected_role:
            error_msg = (
                f"Cannot get data for field '{field.name}' of component '{field.coupled_component.name}'. "
                f"Field has to be a TARGET field, but its exchange_type={field.exchange_type}."
            )
            self._handle_field_validation_error(error_msg)
            return None, CouplerErrorCode.WRONG_EXCHANGE_TYPE

        # Also check the actual YAC role: a field absent from the coupling YAML has role NONE
        # and yac_field.get() would silently return zeros -- signal this via error code so the
        # caller can decide whether to use the YAC fallback or supply their own.
        assert field.yac_field is not None, f"YAC field for '{field.name}' has not been created yet."
        role = self.interface.get_field_role(
            field.yac_field.component_name,
            field.yac_field.grid_name,
            field.yac_field.name,
        )
        if role is not expected_role:
            error_msg = (
                f"Field '{field.name}' is declared TARGET in EBFM but its actual YAC role "
                f"is '{role}' (field not present in coupling config)."
            )
            self._handle_field_validation_error(error_msg)
            error = CouplerErrorCode.WRONG_ROLE

        logger.debug(f"Receiving field {field.name} from {field.coupled_component.name}...")
        data, _ = field.yac_field.get()
        logger.debug(f"Receiving field {field.name} from {field.coupled_component.name} complete.")
        return data[0], error

    def has_field(self, component_name: str, field_name: str, exchange_type) -> bool:
        """
        Check whether a field with given name and exchange type exists for a coupled component.

        @param[in] component_name name of the component
        @param[in] field_name name of the field
        @param[in] exchange_type expected exchange type

        @returns True if such a field exists, otherwise False
        """
        if not self.has_coupling_to(component_name):
            return False

        component = self._coupled_components[component_name]
        fields = self.fields.filter(
            lambda f: f.coupled_component == component and f.name == field_name and f.exchange_type == exchange_type
        )
        return not fields.is_empty()

    def _handle_field_validation_error(self, error_msg: str):
        """
        Handle field validation errors according to the configured validation level.

        @param[in] error_msg error message to log or raise

        @raises RuntimeError if validation level is FATAL
        """
        from ebfm.core.config import FieldValidationLevel

        if self.field_validation_level == FieldValidationLevel.FATAL:
            raise RuntimeError(error_msg)
        elif self.field_validation_level == FieldValidationLevel.WARNING:
            logger.warning(error_msg)
        else:  # SILENT
            logger.debug(error_msg)

    def finalize(self):
        """
        Finalize the coupling interface
        """

        logger.info("Finalizing YAC Coupling...")
        del self.interface
        logger.info("YAC Coupling finalized.")

    def _add_grid(self, grid_name, grid):
        """
        Adds a grid to the Coupler interface.

        @param[in] grid_name name of the grid in YAC
        @param[in] grid Grid object used by EBFM where coupling happens
        """

        assert not self.grid, "Grid has already been added to YACCoupler."

        self.grid = yac.UnstructuredGrid(
            grid_name,
            np.full(len(grid.cell_ids), grid.num_vertices_per_cell),
            grid.lon,
            grid.lat,
            grid.cell_to_vertex.flatten(),
        )

        self.grid.set_global_index(grid.vertex_ids, yac.Location.CORNER)
        self.corner_points = self.grid.def_points(yac.Location.CORNER, grid.lon, grid.lat)

    def _add_couples(self, field_definitions: FieldSet):
        """
        Adds coupling definitions to the Coupler interface.

        @param[in] field_definitions FieldDefinitions object containing field definitions for EBFM
        """
        self._construct_coupling_pre_sync(field_definitions)

        self.interface.sync_def()

        self._construct_coupling_post_sync()

    def _construct_coupling_pre_sync(self, field_definitions: FieldSet):
        """
        Constructs the coupling interface with YAC.

        @param[in] field_definitions FieldDefinitions object containing field definitions for EBFM
        """

        assert self.fields.is_empty(), "Coupling fields have already been constructed."

        collection_size = 1  # TODO: Dummy value for now; make configurable if needed

        for field in field_definitions:
            assert self.has_coupling_to(
                field.coupled_component.name
            ), f"Cannot add field '{field.name}' for uncoupled component '{field.coupled_component.name}'."

            yac_field = field.construct_yac_field(self.interface, self.component, collection_size, self.corner_points)
            self.fields.add(yac_field)

    def _construct_coupling_post_sync(self):
        # after synchronisation or the end of the definition phase YAC can be queried about various information

        for field in self.fields:
            yac_field = field.yac_field
            is_defined = self.interface.get_field_is_defined(
                yac_field.component_name, yac_field.grid_name, yac_field.name
            )
            assert is_defined, (
                f"Field '{yac_field.name}' is not defined in YAC for component '{yac_field.component_name}' and "
                f"grid '{yac_field.grid_name}'."
            )

            field_role = self.interface.get_field_role(yac_field.component_name, yac_field.grid_name, yac_field.name)

            if field_role is yac.ExchangeType.TARGET:
                logger.debug(
                    f"Field {yac_field.name}: "
                    f"SOURCE {field.coupled_component.name} -> TARGET {yac_field.component_name}"
                )
                field_info = field.get_info(self.interface)
                logger.info(field_info)
