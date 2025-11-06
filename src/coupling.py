# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import yac
from pathlib import Path
import numpy as np
from collections import namedtuple
from typing import Dict

from elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?

import logging

logger = logging.getLogger(__name__)


field_template = """
field {name}:
 - source:
   - component: {comp}
   - grid:      {grid}
   - timestep:  {timestep}
   - metadata:  {metadata}
"""


def days_to_iso(days: float) -> str:
    """
    Convert a time step in days to ISO 8601 format.

    @param[in] days time step in days
    @returns ISO 8601 formatted string representing the time step
    """
    import pandas as pd

    dt = pd.Timedelta(days=days)
    return dt.isoformat()


class Coupler:
    component_name: str  # name of this component in the coupler configuration

    couple_to_elmer_ice: bool  # whether to couple with Elmer/Ice
    couple_to_icon_atmo: bool  # whether to couple with ICON atmosphere

    @property
    def has_coupling(self) -> bool:
        return_value = self.couple_to_elmer_ice or self.couple_to_icon_atmo
        logger.debug(f"Component {self.component_name} has_coupling={return_value}")
        return return_value

    def __init__(self, component_name: str, coupler_config: Path = None):
        """
        Create Coupler object

        @param[in] component_name name of this component in the coupler configuration
        @param[in] coupler_config path to the coupler configuration file
        """
        raise NotImplementedError("Coupler is an abstract base class and cannot be instantiated directly.")

    def add_grid(self, grid_name: str, grid: Grid):
        """
        Add grid to the Coupler interface
        """
        raise NotImplementedError("add_grid method must be implemented in subclasses.")

    def add_couples(self, time: Dict[str, float]):
        """
        Add coupling definitions to the Coupler interface
        """
        raise NotImplementedError("add_couples method must be implemented in subclasses.")


class NoCoupler(Coupler):
    couple_to_icon_atmo: bool = False
    couple_to_elmer_ice: bool = False

    def __init__(self, component_name: str):
        self.component_name = component_name
        logger.debug(f"NoCoupler created for component '{component_name}'.")


class YACCoupler(Coupler):
    interface: yac.YAC = None
    component: yac.Component = None
    grid: yac.UnstructuredGrid = None
    corner_points: yac.Points = None
    # TODO use a single dictionary for both source and target fields;
    # TODO provide properties to create dicts containing source_fields and target_fields by comparing to exchange_type
    source_fields: Dict[str, yac.Field] = {}
    target_fields: Dict[str, yac.Field] = {}

    def __init__(self, component_name: str, coupler_config: Path):
        logger.debug(f"YAC version is {yac.version()}")
        self.interface = yac.YAC()
        self.component_name = component_name
        self.interface.read_config_yaml(str(coupler_config))
        self.component = self.interface.def_comp(component_name)

    def add_grid(self, grid_name, grid):
        """
        Adds a grid to the Coupler interface.

        @param[in] grid_name name of the grid in YAC
        @param[in] grid Grid object used by EBFM where coupling happens
        """

        self.grid = yac.UnstructuredGrid(
            grid_name,
            np.full(len(grid.cell_ids), grid.num_vertices_per_cell),
            grid.lon,
            grid.lat,
            grid.cell_to_vertex.flatten(),
        )

        self.grid.set_global_index(grid.vertex_ids, yac.Location.CORNER)
        self.corner_points = self.grid.def_points(yac.Location.CORNER, grid.lon, grid.lat)

    def add_couples(self, time: Dict[str, float]):
        """
        Adds coupling definitions to the Coupler interface.

        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        self.construct_coupling(time)

        self.interface.sync_def()

        self.construct_coupling_post_sync()

        self.interface.enddef()

    def construct_coupling(self, time: Dict[str, float]):
        """
        Constructs the coupling interface with YAC.

        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        if self.couple_to_elmer_ice:
            self.construct_coupling_elmer_ice(time)

        if self.couple_to_icon_atmo:
            self.construct_coupling_icon_atmo(time)

    def construct_coupling_elmer_ice(self, time: Dict[str, float]):
        assert self.couple_to_elmer_ice, "Cannot construct coupling to Elmer/Ice if 'couple_to_elmer_ice' is False."

        FieldDefinition = namedtuple("FieldDefinition", ["exchange_type", "name", "metadata"])
        # TODO: Get hard-coded data below from dummies/EBFM/ebfm-config.yaml
        field_definitions = [
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
        ]

        timestep_value = days_to_iso(time["dt"])
        collection_size = 1  # TODO: Dummy value for now; make configurable if needed

        Timestep = namedtuple("Timestep", ["value", "format"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        for field_definition in field_definitions:
            field = yac.Field.create(
                field_definition.name,
                self.component,
                self.corner_points,
                collection_size,
                timestep.value,
                timestep.format,
            )
            self.interface.def_field_metadata(
                field.component_name,
                field.grid_name,
                field_definition.name,
                field_definition.metadata.encode("utf-8"),
            )

            if field_definition.exchange_type == yac.ExchangeType.SOURCE:
                self.source_fields[field_definition.name] = field

            elif field_definition.exchange_type == yac.ExchangeType.TARGET:
                self.target_fields[field_definition.name] = field

    def construct_coupling_icon_atmo(self, time: Dict[str, float]):
        assert self.couple_to_icon_atmo, "Cannot construct coupling to ICON if 'couple_to_icon_atmo' is False."

        FieldDefinition = namedtuple("FieldDefinition", ["exchange_type", "name", "metadata"])
        # TODO: Get hard-coded data below from dummies/EBFM/ebfm-config.yaml
        field_definitions = [
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
            FieldDefinition(
                exchange_type=yac.ExchangeType.TARGET, name="tas", metadata="Temperature at surface (in K)"
            ),
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
        ]

        timestep_value = days_to_iso(time["dt"])
        collection_size = 1  # TODO: Dummy value for now; make configurable if needed

        Timestep = namedtuple("Timestep", ["value", "format"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        for field_definition in field_definitions:
            field = yac.Field.create(
                field_definition.name,
                self.component,
                self.corner_points,
                collection_size,
                timestep.value,
                timestep.format,
            )
            self.interface.def_field_metadata(
                field.component_name, field.grid_name, field_definition.name, field_definition.metadata.encode("utf-8")
            )

            if field_definition.exchange_type == yac.ExchangeType.SOURCE:
                self.source_fields[field_definition.name] = field

            elif field_definition.exchange_type == yac.ExchangeType.TARGET:
                self.target_fields[field_definition.name] = field

    def construct_coupling_post_sync(self):
        # after synchronisation or the end of the definition phase YAC can be queried about various information

        for field in self.target_fields.values():
            is_defined = self.interface.get_field_is_defined(field.component_name, field.grid_name, field.name)
            assert is_defined, (
                f"Field '{field.name}' is not defined in YAC for component '{field.component_name}' and "
                f"grid '{field.grid_name}'."
            )

            field_role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
            assert (
                field_role == yac.ExchangeType.TARGET
            ), f"Field '{field.name}' is no TARGET as expected but a {field_role}."
            field_info = self.field_information(field)
            logger.info(field_info)

    def field_information(self, field: yac.Field) -> str:
        src_comp, src_grid, src_field = self.interface.get_field_source(
            field.component_name, field.grid_name, field.name
        )
        src_field_timestep = self.interface.get_field_timestep(src_comp, src_grid, src_field)
        src_field_metadata = self.interface.get_field_metadata(src_comp, src_grid, src_field)
        return field_template.format(
            name=field.name, comp=src_comp, grid=src_grid, timestep=src_field_timestep, metadata=src_field_metadata
        )

    def _put(self, field_name: str, field_data: np.array):
        field: yac.Field = self.source_fields[field_name]
        field.put(field_data)

    def _get(self, field_name: str) -> np.array:
        field: yac.Field = self.target_fields[field_name]
        data, action = field.get()
        return data

    def exchange_icon_atmo(self, put_data: Dict[str, np.array]) -> Dict[str, np.array]:
        """Exchange data with ICON atmosphere component

        @param[in] put_data dictionary of field names and their data to be exchanged with ICON atmosphere component

        @returns dictionary of exchanged field data
        """
        assert self.couple_to_icon_atmo, "Cannot exchange data with ICON atmosphere if 'couple_to_icon_atmo' is False."

        icon_fields = set(
            [
                # 'albedo',
                "pr",
                "pr_snow",
                "rsds",
                "rlds",
                "clt",
                "sfcwind",
                "tas",
                # 'huss',
                # 'sfcPressure'
            ]
        )

        for field_name, field in self.source_fields.items():
            logger.debug(f"Checking field {field_name} for ICON atmosphere exchange...")
            if field_name in icon_fields:
                role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
                assert (
                    role == yac.ExchangeType.SOURCE
                ), f"Field '{field_name}' is not a target field for ICON exchange, but has role '{role}'."
                logger.debug(f"Sending field {field_name} to ICON atmosphere...")
                self._put(field_name, put_data[field_name])
            else:
                logger.debug(f"Skipping field {field_name} as it is not part of ICON atmosphere exchange.")

        received_data = {}
        for field_name, field in self.target_fields.items():
            logger.debug(f"Checking field {field_name} for ICON atmosphere exchange...")
            if field_name in icon_fields:
                role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
                assert (
                    role == yac.ExchangeType.TARGET
                ), f"Field '{field_name}' is not a target field for ICON exchange, but has role '{role}'."
                logger.debug(f"Receiving field {field_name} from ICON atmosphere...")
                # Only get the first element, since we only have collection size of 1
                received_data[field_name] = self._get(field_name)[0]
            else:
                logger.debug(f"Skipping field {field_name} as it is not part of ICON atmosphere exchange.")

        return received_data

    def exchange_elmer_ice(self, put_data: Dict[str, np.array]) -> Dict[str, np.array]:
        """Exchange data with Elmer Ice component

        @param[in] put_data dictionary of field names and their data to be exchanged with Elmer Ice component
        @param[in] get_keys list of field names to be received from Elmer Ice component

        @returns dictionary of exchanged field data
        """
        assert self.couple_to_elmer_ice, "Cannot exchange data with Elmer/Ice if 'couple_to_elmer_ice' is False."

        # TODO: make configurable if needed
        elmer_fields = set(
            [
                "smb",
                "T_ice",
                "runoff",
                "h",
                # 'dhdx',
                # 'dhdy',
            ]
        )

        for field_name, field in self.source_fields.items():
            if field_name in elmer_fields:
                role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
                assert (
                    role == yac.ExchangeType.SOURCE
                ), f"Field '{field_name}' is not a target field for Elmer/Ice exchange, but has role '{role}'."
                self._put(field_name, put_data[field_name])

        received_data = {}
        for field_name, field in self.target_fields.items():
            if field_name in elmer_fields:
                role = self.interface.get_field_role(field.component_name, field.grid_name, field.name)
                assert (
                    role == yac.ExchangeType.TARGET
                ), f"Field '{field_name}' is not a target field for Elmer/Ice exchange, but has role '{role}'."
                # Only get the first element, since we only have collection size of 1
                received_data[field_name] = self._get(field_name)[0]

        return received_data

    def finalize(self):
        """Finalize the coupling interface"""
        del self.interface


def init(
    coupler_config: Path, ebfm_coupling_config: Path, couple_with_icon_atmo: bool, couple_with_elmer_ice: bool
) -> Coupler:
    """Create interface to the coupler and register component

    @param[in] path to global Coupler configuration file
    @param[in] ebfm_coupling_config path to local configuration of coupling for EBFM
    @param[in] couple_with_icon_atmo whether to couple with ICON atmosphere
    @param[in] couple_with_elmer_ice whether to couple with Elmer/Ice

    @returns Coupler object
    """

    # TODO: use ebfm_coupling_config for EBFM specific configuration?

    component_name = "ebfm"  # TODO: get from ebfm_coupling_config?
    coupler = YACCoupler(component_name, coupler_config)

    coupler.couple_to_icon_atmo = couple_with_icon_atmo
    coupler.couple_to_elmer_ice = couple_with_elmer_ice

    # TODO: yac_cget_comp_comm(comp_id, elmer_comm) needed?

    return coupler


def setup(coupler: Coupler, grid: Grid, time: Dict[str, float]):
    """Performs initialization operations after init and before enterint the
    time loop

    @param[in] coupler Coupler object with interface to YAC
    @param[in] grid Grid used by EBFM where coupling happens
    @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
    """
    grid_name = "ebfm_grid"  # TODO: get from ebfm_coupling_config?

    coupler.add_grid(grid_name, grid)
    coupler.add_couples(time)


def finalize(coupler: Coupler):
    logger.debug("Finalizing coupling...")
    coupler.finalize()
    logger.info("Coupling finalized.")
