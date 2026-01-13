# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import yac
import numpy as np

from ebfm import logging

from typing import Set, Callable

from coupling.couplers.base import Coupler, Grid, Dict, CouplingConfig

# from coupling import Field  # TODO: rather use generic Field from coupling
from coupling.couplers.yacField import Field

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?

logger = logging.getLogger(__name__)


field_template = """
field {name}:
 - source:
   - component: {comp}
   - grid:      {grid}
   - timestep:  {timestep}
   - metadata:  {metadata}
"""


class FieldSet:
    """
    Set of fields.

    Can be used to collect fields and perform filtering operations for components, exchange types, etc.

    Example:
        fields = FieldSet()
        fields.add(Field(..., exchange_type=yac.ExchangeType.SOURCE))
        fields.add(Field(..., exchange_type=yac.ExchangeType.TARGET))
        source_fields = fields.filter(lambda f: f.exchange_type == yac.ExchangeType.SOURCE)
    """

    def __init__(self, fields: Set[Field] = None):
        """
        Initialize FieldSet.
        """
        self._fields = fields if fields is not None else set()

    def __iter__(self):
        return iter(self._fields)

    def is_empty(self) -> bool:
        return len(self._fields) == 0

    def all(self) -> Set[Field]:
        return set(self._fields)

    def filter(self, condition: Callable[[Field], bool]) -> "FieldSet":
        return FieldSet(set(d for d in self._fields if condition(d)))

    def add(self, field: Field):
        assert field not in self._fields, f"Field {field} with name {field.name} already exists in FieldSet."
        self._fields.add(field)


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

        field_definitions = set()

        for component in self._coupled_components.values():
            field_definitions |= component.get_field_definitions(time)

        self._add_couples(FieldSet(field_definitions))

    def _get_field(self, component_name: str, field_name: str) -> Field:
        """
        Get Field object for given component and field name

        @param[in] component_name name of the component
        @param[in] field_name name of the field

        @returns Field object
        """

        assert self.has_coupling_to(
            component_name
        ), f"Cannot get field for {component_name} because no coupling exists."

        component = self._coupled_components[component_name]

        comp_fields = self.fields.filter(lambda f: f.coupled_component == component and f.name == field_name).all()

        assert (
            len(comp_fields) == 1
        ), f"Expected exactly one field for '{field_name}' from component '{component_name}', "

        return comp_fields.pop()

    def put(self, component_name: str, field_name: str, data: np.array):
        """
        Put data to another component

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data to be sent
        """

        field = self._get_field(component_name, field_name)

        assert (
            field.exchange_type == yac.ExchangeType.SOURCE
        ), f"Cannot put data for field '{field.name}' of component '{field.coupled_component.name}'. "
        f"Field has to be a SOURCE field, but its '{field.exchange_type=}'."

        logger.debug(f"Sending field {field.name} to {field.coupled_component.name}...")
        field.yac_field.put(data)
        logger.debug(f"Sending field {field.name} to {field.coupled_component.name} complete.")

    def get(self, component_name: str, field_name: str) -> np.array:
        """
        Get data from another component

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to get data for

        @returns field data
        """

        field = self._get_field(component_name, field_name)

        assert (
            field.exchange_type == yac.ExchangeType.TARGET
        ), f"Cannot get data for field '{field.name}' of component '{field.coupled_component.name}'. "
        f"Field has to be a TARGET field, but its '{field.exchange_type=}'."

        logger.debug(f"Receiving field {field.name} from {field.coupled_component.name}...")
        data, _ = field.yac_field.get()
        logger.debug(f"Receiving field {field.name} from {field.coupled_component.name} complete.")
        return data[0]

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

        self.interface.enddef()

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
                field_info = self._field_information(field)
                logger.info(field_info)

    def _field_information(self, field: Field) -> str:
        """
        Get detailed information about a Field.

        @param[in] field yac.Field to get information about
        @returns Formatted string with field information
        """
        assert field.yac_field, f"YAC field is not defined for field {field}."

        src_comp, src_grid, src_field = self.interface.get_field_source(
            field.yac_field.component_name, field.yac_field.grid_name, field.yac_field.name
        )
        src_field_timestep = self.interface.get_field_timestep(src_comp, src_grid, src_field)
        src_field_metadata = self.interface.get_field_metadata(src_comp, src_grid, src_field)
        assert (
            field.yac_field.name == field.name
        ), f"Field name mismatch: expected '{field.name}', got '{field.yac_field.name}'."
        return field_template.format(
            name=field.name, comp=src_comp, grid=src_grid, timestep=src_field_timestep, metadata=src_field_metadata
        )
