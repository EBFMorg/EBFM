# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import yac
import numpy as np
from dataclasses import dataclass, replace

from ebfm import logging

from typing import Set, Callable
from couplers.base import Coupler, Grid, Dict, CouplingConfig, Component

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


@dataclass(frozen=True)
class Timestep:
    value: str  # value of the timestep in specified format
    format: yac.TimeUnit  # format of the timestep value


@dataclass(frozen=True)
class Field:
    """
    Object for definition of a field to be exchanged via YAC.
    """

    name: str  # name of the field
    component: Component  # component this field belongs to
    timestep: Timestep  # timestep of the field
    metadata: str = None  # optional to allow model providing metadata
    exchange_type: yac.ExchangeType = None  # optional for consistency checks by model configuration
    yac_field: yac.Field = None  # optional if YAC field has been created

    def construct_yac_field(
        self, yac_interface: yac.YAC, yac_component: yac.Component, collection_size: int, corner_points: yac.Points
    ) -> "Field":
        """
        Create a new Field instance with the provided YAC field.

        @param[in] yac_interface handle to YAC interface
        @param[in] yac_component handle to YAC component object
        @param[in] collection_size size of the collection for this field
        @param[in] corner_points yac.Points of the grid for this field

        @returns New Field instance with the provided YAC field
        """
        assert not self.yac_field, f"Field '{self.name}' for component '{self.name}' has already been created in YAC."

        yac_field = yac.Field.create(
            self.name,
            yac_component,
            corner_points,
            collection_size,
            self.timestep.value,
            self.timestep.format,
        )

        # add optional metadata
        if self.metadata:
            yac_interface.def_field_metadata(
                yac_field.component_name,
                yac_field.grid_name,
                yac_field.name,
                self.metadata.encode("utf-8"),
            )

        # perform optional consistency check
        if self.exchange_type:
            field_role = yac_interface.get_field_role(yac_field.component_name, yac_field.grid_name, yac_field.name)
            assert field_role == self.exchange_type, (
                f"Field '{self.name}' role mismatch: expected '{self.exchange_type}', " f"got '{field_role}'."
            )

        return replace(self, yac_field=yac_field)


class FieldCollection:
    """
    Collection of fields.

    Can be used to collect fields and perform filtering operations for components, exchange types, etc.

    Example:
        fields = FieldCollection()
        fields.add(Field(..., exchange_type=yac.ExchangeType.SOURCE))
        fields.add(Field(..., exchange_type=yac.ExchangeType.TARGET))
        source_fields = fields.filter(lambda f: f.exchange_type == yac.ExchangeType.SOURCE)
    """

    def __init__(self, fields: Set[Field] = None):
        """
        Initialize FieldCollection.
        """
        self._fields = fields if fields is not None else set()

    def __iter__(self):
        return iter(self._fields)

    def is_empty(self) -> bool:
        return len(self._fields) == 0

    def all(self) -> Set[Field]:
        return set(self._fields)

    def filter(self, condition: Callable[[Field], bool]) -> "FieldCollection":
        return FieldCollection(set(d for d in self._fields if condition(d)))

    def add(self, field: Field):
        assert field not in self._fields, f"Field {field} with name {field.name} already exists in FieldCollection."
        self._fields.add(field)


# TODO: Get hard-coded data below from dummies/EBFM/ebfm-config.yaml
def get_field_definitions(time: Dict[str, float]) -> Set[Field]:
    """
    Get field definitions for EBFM coupling.

    @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
    """

    timestep_value = days_to_iso(time["dt"])
    timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

    return {
        # Field(
        #     name="T_ice",
        #     component=Component.elmer_ice,
        #     timestep=timestep,
        #     metadata="Near surface temperature at Ice surface (in K)",
        #     exchange_type=yac.ExchangeType.SOURCE,
        # ),
        Field(
            name="smb",
            component=Component.elmer_ice,
            timestep=timestep,
            metadata="??? (in ???)",
            exchange_type=yac.ExchangeType.SOURCE,
        ),
        # Field(
        #     name="runoff",
        #     component=Component.elmer_ice,
        #     timestep=timestep,
        #     metadata="Runoff (in ???)",
        #     exchange_type=yac.ExchangeType.SOURCE,
        # ),
        # Field(
        #     name="h",
        #     component=Component.elmer_ice,
        #     timestep=timestep,
        #     metadata="Surface height (in m)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="dhdx",
        #     component=Component.elmer_ice,
        #     timestep=timestep,
        #     metadata="Surface slope in x direction",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="dhdy",
        #     component=Component.elmer_ice,
        #     timestep=timestep,
        #     metadata="Surface slope in y direction",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="albedo",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Albedo of the ice surface (in ???)",
        #     exchange_type=yac.ExchangeType.SOURCE,
        # ),
        # Field(
        #     name="pr",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Precipitation rate (in kg m-2 s-1)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="pr_snow",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Precipitation rate of snow (in kg m-2 s-1)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="rsds",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Downward shortwave radiation flux (in W m-2)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="rlds",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Downward longwave radiation flux (in W m-2)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="sfcwind",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Wind speed at surface (in m s-1)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="clt",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Cloud cover (in fraction)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="tas",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Temperature at surface (in K)",
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="huss",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Specific humidity at surface (in kg kg-1)"
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
        # Field(
        #     name="sfcPressure",
        #     component=Component.icon_atmo,
        #     timestep=timestep,
        #     metadata="Surface pressure (in Pa)"
        #     exchange_type=yac.ExchangeType.TARGET,
        # ),
    }


def days_to_iso(days: float) -> str:
    """
    Convert a time step in days to ISO 8601 format.

    @param[in] days time step in days
    @returns ISO 8601 formatted string representing the time step
    """
    import pandas as pd

    dt = pd.Timedelta(days=days)
    return dt.isoformat()


class YACCoupler(Coupler):
    interface: yac.YAC = None
    component: yac.Component = None
    grid: yac.UnstructuredGrid = None
    corner_points: yac.Points = None
    fields: FieldCollection = FieldCollection()

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create interface to the coupler and register component

        @param[in] coupling_config coupling configuration of this component
        """

        logger.debug(f"YAC version is {yac.version()}")
        self.interface = yac.YAC()
        self.component_name = coupling_config.component_name

        if coupling_config.coupler_config:
            self.interface.read_config_yaml(str(coupling_config.coupler_config))

        self.component = self.interface.def_comp(self.component_name)
        self._couples_to[Component.icon_atmo] = coupling_config.couple_to_icon_atmo
        self._couples_to[Component.elmer_ice] = coupling_config.couple_to_elmer_ice

    def setup(self, grid: Grid, time: Dict[str, float]):
        """
        Setup the coupling interface

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        grid_name = "ebfm_grid"  # TODO: get from ebfm_coupling_config?

        self._add_grid(grid_name, grid)

        field_definitions = get_field_definitions(time)

        self._add_couples(FieldCollection(field_definitions))

    def exchange(self, component_name: str, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with component

        @param[in] data_to_exchange dictionary of field names and their data to be exchanged with component

        @returns dictionary of exchanged field data
        """

        component = Component[component_name]

        assert self._couples_to[
            component
        ], f"Cannot exchange data with {component=} because {self._couples_to[component]=}'."

        comp_fields = self.fields.filter(lambda f: f.component == component)

        for field in comp_fields.filter(lambda f: f.exchange_type == yac.ExchangeType.SOURCE):
            logger.debug(f"Sending field {field.name} to {component.name}...")
            field.yac_field.put(data_to_exchange[field.name])
            logger.debug(f"Sending field {field.name} to {component.name} complete.")

        received_data = {}

        for field in comp_fields.filter(lambda f: f.exchange_type == yac.ExchangeType.TARGET):
            logger.debug(f"Receiving field {field.name} to {component.name}...")
            data, _ = field.yac_field.get()
            received_data[field.name] = data[0]
            logger.debug(f"Receiving field {field.name} to {component.name} complete.")

        return received_data

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

    def _add_couples(self, field_definitions: FieldCollection):
        """
        Adds coupling definitions to the Coupler interface.

        @param[in] field_definitions FieldDefinitions object containing field definitions for EBFM
        """
        self._construct_coupling_pre_sync(field_definitions)

        self.interface.sync_def()

        self._construct_coupling_post_sync()

        self.interface.enddef()

    def _construct_coupling_pre_sync(self, field_definitions: FieldCollection):
        """
        Constructs the coupling interface with YAC.

        @param[in] field_definitions FieldDefinitions object containing field definitions for EBFM
        """

        assert self.fields.is_empty(), "Coupling fields have already been constructed."

        collection_size = 1  # TODO: Dummy value for now; make configurable if needed

        for field in field_definitions:
            assert self._couples_to[
                field.component
            ], f"Cannot add field '{field.name}' for uncoupled component '{field.component.name}'."

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
                    f"Field {yac_field.name}: SOURCE {field.component.name} -> TARGET {yac_field.component_name}"
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
