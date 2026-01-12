# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from coupling.components import Component
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


@dataclass(frozen=True)
class Field:
    """
    Object for definition of a field to be exchanged via YAC.
    """

    name: str  # name of the field
    # TODO: remove coupler_component and directly store fields in coupling.components.Component?
    coupled_component: Component  # component this field couples to
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
