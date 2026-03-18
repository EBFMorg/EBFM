# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from .base import Field, ExchangeType, Timestep
from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING, Dict

import yac

if TYPE_CHECKING:
    from ebfm.coupling.components.base import Component


@dataclass(frozen=True)
class YACTimestep:
    value: str  # value of the timestep in specified format
    format: yac.TimeUnit  # format of the timestep value

    @classmethod
    def from_timestep(cls, timestep: Timestep) -> "YACTimestep":
        """
        Create a YACTimestep from a generic Timestep.

        Converts the generic representation to a YAC compatible ISO-8601 duration.
        Raises ValueError if conversion is not possible.

        @param[in] timestep generic timestep
        @returns YAC timestep in ISO format
        """
        import pandas as pd

        assert isinstance(timestep, Timestep), f"Expected generic Timestep, got {type(timestep)}"

        try:
            normalized_iso = pd.Timedelta(timestep.value).isoformat()
        except Exception as exc:
            raise ValueError(f"Timestep '{timestep.value}' is not compatible with YAC ISO_FORMAT conversion.") from exc

        return cls(value=normalized_iso, format=yac.TimeUnit.ISO_FORMAT)


field_template = """
field {name}:
 - source:
   - component: {comp}
   - grid:      {grid}
   - timestep:  {timestep}
   - metadata:  {metadata}
"""

EXCHANGE_TYPE_MAP: Dict[ExchangeType, yac.ExchangeType] = {
    ExchangeType.SOURCE: yac.ExchangeType.SOURCE,
    ExchangeType.TARGET: yac.ExchangeType.TARGET,
}


@dataclass(frozen=True)
class YACField:
    """
    Object for definition of a field to be exchanged via YAC.
    """

    name: str
    coupled_component: "Component"
    timestep: YACTimestep  # YAC-specific timestep representation
    exchange_type: yac.ExchangeType  # YAC-specific exchange role
    metadata: Optional[str] = None
    field_handle: Optional[yac.Field] = None  # optional if YAC field has been created

    @staticmethod
    def map_exchange_type(exchange_type: ExchangeType) -> yac.ExchangeType:
        """
        Map generic ExchangeType to YAC exchange type.
        """
        try:
            return EXCHANGE_TYPE_MAP[exchange_type]
        except KeyError as exc:
            raise AssertionError(
                f"Unsupported generic exchange type '{exchange_type}' for YAC coupler. "
                f"Supported exchange types are: {list(EXCHANGE_TYPE_MAP.keys())}."
            ) from exc

    @classmethod
    def from_field(cls, field: Field) -> "YACField":
        """
        Create a YACField from a generic Field definition.

        @param[in] field generic field definition

        @returns YACField instance with mapped YAC exchange type and ISO timestep
        """

        yac_exchange_type = cls.map_exchange_type(field.exchange_type)

        timestep = YACTimestep.from_timestep(field.timestep)

        return cls(
            name=field.name,
            coupled_component=field.coupled_component,
            timestep=timestep,
            exchange_type=yac_exchange_type,
            metadata=field.metadata,
        )

    def construct_yac_field(
        self, yac_interface: yac.YAC, yac_component: yac.Component, collection_size: int, corner_points: yac.Points
    ) -> "YACField":
        """
        Create a new Field instance with the provided YAC field.

        @param[in] yac_interface handle to YAC interface
        @param[in] yac_component handle to YAC component object
        @param[in] collection_size size of the collection for this field
        @param[in] corner_points yac.Points of the grid for this field

        @returns New Field instance with the provided YAC field
        """
        assert (
            not self.field_handle
        ), f"Field '{self.name}' for component '{self.name}' has already been created in YAC."

        # TODO: work-around since some components assume that metadata is always set, components should actually check
        #       for existence of metadata and only call yac_cget_field_metadata or yac_fget_field_metadata if metadata
        #       exists.
        metadata = self.metadata or "N/A"

        yac_field = yac.Field.create(
            self.name,
            yac_component,
            corner_points,
            collection_size,
            self.timestep.value,
            self.timestep.format,
        )

        # add optional metadata
        if metadata:
            yac_interface.def_field_metadata(
                yac_field.component_name,
                yac_field.grid_name,
                yac_field.name,
                metadata.encode("utf-8"),
            )

        return replace(self, field_handle=yac_field)

    def perform_consistency_checks(self, yac_interface: yac.YAC, field_validation_level=None):
        """
        Perform consistency checks on the YACField.

        Ensures that self.yac_field (inside YAC) is consistent with the Field attributes stored here.

        @note should be called after enddef since this is the point where we can guarantee that YAC has all information.

        @param[in] field_validation_level optional FieldValidationLevel controlling how role mismatches are handled.
                                          If None (or FATAL), a role mismatch raises AssertionError.
                                          If WARNING or SILENT, a mismatch is only logged.
        """
        from ebfm.core.config import FieldValidationLevel

        assert self.field_handle is not None, f"YAC field for '{self.name}' has not been created yet."
        assert self.field_handle.component_name == "ebfm", (
            f"Field '{self.name}' coupled component '{self.coupled_component.name}' does not match "
            f"YAC component '{self.field_handle.component_name}'."
        )
        assert (
            self.name == self.field_handle.name
        ), f"Field '{self.name}' name does not match YAC field name '{self.field_handle.name}'."
        assert self.exchange_type in (yac.ExchangeType.SOURCE, yac.ExchangeType.TARGET), (
            f"Field '{self.name}' has invalid YAC exchange type '{self.exchange_type}'. "
            "Must be either SOURCE or TARGET."
        )
        field_role = yac_interface.get_field_role(
            self.field_handle.component_name, self.field_handle.grid_name, self.field_handle.name
        )
        if field_role != self.exchange_type:
            msg = f"Field '{self.name}' role mismatch: expected '{self.exchange_type}', got '{field_role}'."
            if field_validation_level is None or field_validation_level == FieldValidationLevel.FATAL:
                raise AssertionError(msg)
            elif field_validation_level == FieldValidationLevel.WARNING:
                import logging as _logging

                _logging.getLogger(__name__).warning(msg)
            else:  # SILENT
                import logging as _logging

                _logging.getLogger(__name__).debug(msg)

    def get_info(self, yac_interface: yac.YAC) -> str:
        """
        Get detailed information about a Field.

        @param[in] field yac.Field to get information about
        @returns Formatted string with field information
        """

        assert self.field_handle, f"YAC field is not defined for field {self}."

        src_comp, src_grid, src_field = yac_interface.get_field_source(
            self.field_handle.component_name, self.field_handle.grid_name, self.field_handle.name
        )
        src_field_timestep = yac_interface.get_field_timestep(src_comp, src_grid, src_field)

        if self.metadata:  # metadata is optional
            src_field_metadata = yac_interface.get_field_metadata(src_comp, src_grid, src_field)
        else:
            src_field_metadata = "N/A"

        assert (
            self.field_handle.name == self.name
        ), f"Field name mismatch: expected '{self.name}', got '{self.field_handle.name}'."
        return field_template.format(
            name=self.name, comp=src_comp, grid=src_grid, timestep=src_field_timestep, metadata=src_field_metadata
        )
