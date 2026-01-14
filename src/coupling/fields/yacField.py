# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from coupling.fields.base import Field
from dataclasses import dataclass, replace
import yac  # should not be needed here. Maybe consider actually having a YACField inherit from Field?


@dataclass(frozen=True)
class Timestep:
    value: str  # value of the timestep in specified format
    format: yac.TimeUnit  # format of the timestep value


def days_to_iso(days: float) -> str:
    """
    Convert a time step in days to ISO 8601 format.

    @param[in] days time step in days
    @returns ISO 8601 formatted string representing the time step
    """
    import pandas as pd

    dt = pd.Timedelta(days=days)
    return dt.isoformat()


field_template = """
field {name}:
 - source:
   - component: {comp}
   - grid:      {grid}
   - timestep:  {timestep}
   - metadata:  {metadata}
"""


@dataclass(frozen=True)
class YACField(Field):
    """
    Object for definition of a field to be exchanged via YAC.
    """

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

        # TODO: work-around since some components assume that metadata is always set, components should actually check
        #       for existence of metadata and only call yac_cget_field_metadata or yac_fget_field_metadata if metadata
        #       exists.
        if not self.metadata:
            self = replace(self, metadata="N/A")

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

        return replace(self, yac_field=yac_field)

    def perform_consistency_checks(self, yac_interface: yac.YAC):
        """
        Perform consistency checks on the YACField.

        Ensures that self.yac_field (inside YAC) is consistent with the Field attributes stored here.

        @note should be called after enddef since this is the point where we can guarantee that YAC has all information.
        """

        assert self.yac_field is not None, f"YAC field for '{self.name}' has not been created yet."
        assert self.yac_field.component_name == "ebfm", (
            f"Field '{self.name}' coupled component '{self.coupled_component.name}' does not match "
            f"YAC component '{self.yac_field.component_name}'."
        )
        assert (
            self.name == self.yac_field.name
        ), f"Field '{self.name}' name does not match YAC field name '{self.yac_field.name}'."
        assert self.exchange_type in (yac.ExchangeType.SOURCE, yac.ExchangeType.TARGET), (
            f"Field '{self.name}' has invalid exchange type '{self.exchange_type}'. " "Must be either SOURCE or TARGET."
        )
        field_role = yac_interface.get_field_role(
            self.yac_field.component_name, self.yac_field.grid_name, self.yac_field.name
        )
        assert field_role == self.exchange_type, (
            f"Field '{self.name}' role mismatch: expected '{self.exchange_type}', " f"got '{field_role}'."
        )

    def get_info(self, yac_interface: yac.YAC) -> str:
        """
        Get detailed information about a Field.

        @param[in] field yac.Field to get information about
        @returns Formatted string with field information
        """

        assert self.yac_field, f"YAC field is not defined for field {self}."

        src_comp, src_grid, src_field = yac_interface.get_field_source(
            self.yac_field.component_name, self.yac_field.grid_name, self.yac_field.name
        )
        src_field_timestep = yac_interface.get_field_timestep(src_comp, src_grid, src_field)

        if self.metadata:  # metadata is optional
            src_field_metadata = yac_interface.get_field_metadata(src_comp, src_grid, src_field)
        else:
            src_field_metadata = "N/A"

        assert (
            self.yac_field.name == self.name
        ), f"Field name mismatch: expected '{self.name}', got '{self.yac_field.name}'."
        return field_template.format(
            name=self.name, comp=src_comp, grid=src_grid, timestep=src_field_timestep, metadata=src_field_metadata
        )
