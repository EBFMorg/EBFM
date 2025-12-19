# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import yac
import numpy as np
from dataclasses import dataclass

from ebfm import logging

from typing import List
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


@dataclass
class FieldDefinition:
    name: str  # name of the field
    exchange_type: yac.ExchangeType = None  # optional for consistency checks by model configuration
    metadata: str = None  # optional to allow model providing metadata


@dataclass
class Timestep:
    value: str  # value of the timestep in specified format
    format: yac.TimeUnit  # format of the timestep value


# TODO: Get hard-coded data below from dummies/EBFM/ebfm-config.yaml
# TODO: Move Timestep inside FieldDefinition (would allow having different timesteps per field)
all_field_definitions = {
    Component.elmer_ice: [
        FieldDefinition(
            exchange_type=yac.ExchangeType.SOURCE,
            name="T_ice",
            metadata="Near surface temperature at Ice surface (in K)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.SOURCE,
            name="smb",
            metadata="??? (in ???)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.SOURCE,
            name="runoff",
            metadata="Runoff (in ???)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET,
            name="h",
            metadata="Surface height (in m)",
        ),
        # FieldDefinition(
        #     exchange_type=yac.ExchangeType.TARGET,
        #     name="dhdx",
        #     metadata="Surface slope in x direction",
        # ),
        # FieldDefinition(
        #     exchange_type=yac.ExchangeType.TARGET,
        #     name="dhdy",
        #     metadata="Surface slope in y direction",
        # ),
    ],
    Component.icon_atmo: [
        # FieldDefinition(
        #     exchange_type=yac.ExchangeType.SOURCE,
        #     name="albedo",
        #     metadata="Albedo of the ice surface (in ???)"
        # ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET,
            name="pr",
            metadata="Precipitation rate (in kg m-2 s-1)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET,
            name="pr_snow",
            metadata="Precipitation rate of snow (in kg m-2 s-1)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET,
            name="rsds",
            metadata="Downward shortwave radiation flux (in W m-2)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET,
            name="rlds",
            metadata="Downward longwave radiation flux (in W m-2)",
        ),
        FieldDefinition(
            exchange_type=yac.ExchangeType.TARGET, name="sfcwind", metadata="Wind speed at surface (in m s-1)"
        ),
        FieldDefinition(exchange_type=yac.ExchangeType.TARGET, name="clt", metadata="Cloud cover (in fraction)"),
        FieldDefinition(exchange_type=yac.ExchangeType.TARGET, name="tas", metadata="Temperature at surface (in K)"),
        # FieldDefinition(
        #     exchange_type=yac.ExchangeType.TARGET,
        #     name="huss",
        #     metadata="Specific humidity at surface (in kg kg-1)"
        # ),
        # FieldDefinition(
        #     exchange_type=yac.ExchangeType.TARGET,
        #     name="sfcPressure",
        #     metadata="Surface pressure (in Pa)"
        # ),
    ],
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
    fields: Dict[Component, yac.Field] = {}

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

    def setup(self, grid: Dict | Grid, time: Dict[str, float]):
        """
        Setup the coupling interface

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        grid_name = "ebfm_grid"  # TODO: get from ebfm_coupling_config?

        self._add_grid(grid_name, grid)

        timestep_value = days_to_iso(time["dt"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        self._add_couples(timestep)

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

        fields: List[yac.Field] = self.fields[component]

        received_data = {}
        put_fields = []
        get_fields = []

        for field in fields:
            field_role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
            if field_role is yac.ExchangeType.SOURCE:
                put_fields.append(field)
            elif field_role is yac.ExchangeType.TARGET:
                get_fields.append(field)
            else:
                raise Exception(f"Unexpected {field.exchange_type=}")

        for field in put_fields:
            logger.debug(f"Sending field {field.name} to {component.name}...")
            field.put(data_to_exchange[field.name])
            logger.debug(f"Sending field {field.name} to {component.name} complete.")

        for field in get_fields:
            logger.debug(f"Receiving field {field.name} to {component.name}...")
            data, _ = field.get()
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

    def _add_couples(self, timestep: Timestep):
        """
        Adds coupling definitions to the Coupler interface.

        @param[in] timestep Timestep object representing the timestep size of the fields
        """
        self._construct_coupling_pre_sync(timestep)

        self.interface.sync_def()

        self._construct_coupling_post_sync()

        self.interface.enddef()

    def _construct_coupling_pre_sync(self, timestep: Timestep):
        """
        Constructs the coupling interface with YAC.

        @param[in] timestep Timestep object representing the timestep size of the fields
        """

        assert self.fields == {}, "Coupling fields have already been constructed."

        for component, is_coupled in self._couples_to.items():
            if is_coupled:
                assert self.fields.get(component) is None, f"Coupling to {component=} has already been constructed."
                self.fields[component] = self._construct_coupling_to(component, timestep)

    def _construct_coupling_to(self, component: Component, timestep: Timestep) -> List[yac.Field]:
        """
        Constructs coupling fields to a specific Component.

        @param[in] component component to construct coupling to
        @param[in] timestep Timestep object representing the timestep size of the field

        @returns List of yac.Field objects representing the coupling fields
        """

        assert self._couples_to[
            component
        ], f"Cannot construct coupling to {component=} because {self._couples_to[component]=}'."

        collection_size = 1  # TODO: Dummy value for now; make configurable if needed

        fields = list()

        for field_definition in all_field_definitions[component]:
            field = yac.Field.create(
                field_definition.name,
                self.component,
                self.corner_points,
                collection_size,
                timestep.value,
                timestep.format,
            )

            # add optional metadata
            if field_definition.metadata:
                self.interface.def_field_metadata(
                    field.component_name,
                    field.grid_name,
                    field_definition.name,
                    field_definition.metadata.encode("utf-8"),
                )

            # perform optional consistency check
            if field_definition.exchange_type:
                field_role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
                assert field_role == field_definition.exchange_type, (
                    f"Field '{field_definition.name}' role mismatch: expected '{field_definition.exchange_type}', "
                    f"got '{field_role}'."
                )

            fields.append(field)

        return fields

    def _construct_coupling_post_sync(self):
        # after synchronisation or the end of the definition phase YAC can be queried about various information

        for component, fields in self.fields.items():
            for field in fields:
                is_defined = self.interface.get_field_is_defined(field.component_name, field.grid_name, field.name)
                assert is_defined, (
                    f"Field '{field.name}' is not defined in YAC for component '{field.component_name}' and "
                    f"grid '{field.grid_name}'."
                )

                field_role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)

                if field_role is yac.ExchangeType.TARGET:
                    logger.debug(f"Field {field.name}: SOURCE {component.name} -> TARGET {field.component_name}")
                    field_info = self._field_information(field)
                    logger.info(field_info)

    def _field_information(self, field: yac.Field) -> str:
        """
        Get detailed information about a yac.Field.

        @param[in] field yac.Field to get information about
        @returns Formatted string with field information
        """

        src_comp, src_grid, src_field = self.interface.get_field_source(
            field.component_name, field.grid_name, field.name
        )
        src_field_timestep = self.interface.get_field_timestep(src_comp, src_grid, src_field)
        src_field_metadata = self.interface.get_field_metadata(src_comp, src_grid, src_field)
        return field_template.format(
            name=field.name, comp=src_comp, grid=src_grid, timestep=src_field_timestep, metadata=src_field_metadata
        )
