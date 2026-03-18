# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from .base import Field, ExchangeType, Timestep
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeField:
    """
    Object for definition of a fake field for testing purposes.
    """

    name: str
    timestep: Timestep
    exchange_type: ExchangeType

    @classmethod
    def from_field(cls, field: Field) -> "FakeField":
        """
        Create a FakeField from a generic Field definition.

        @param[in] field generic field definition

        @returns FakeField instance
        """

        return cls(
            name=field.name,
            timestep=field.timestep,
            exchange_type=field.exchange_type,
        )
