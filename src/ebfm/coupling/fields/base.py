# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from ebfm.coupling.components.base import Component
from dataclasses import dataclass
from collections.abc import Callable
from enum import Enum


class ExchangeType(Enum):
    """
    Generic field exchange role independent of a specific coupling backend.
    """

    SOURCE = "source"
    TARGET = "target"


GenericExchangeType = ExchangeType
"""Alias clarifying that `ExchangeType` is the backend-independent exchange role type."""


@dataclass(frozen=True)
class Timestep:
    """
    Generic timestep wrapper.

    For now this stores a single string value (typically ISO-8601 duration),
    but can be extended later to wrap datetime/timedelta types.
    """

    value: str


@dataclass(frozen=True)
class Field:
    """
    Object for definition of a generic field.
    """

    name: str  # name of the field
    # TODO: remove coupler_component and directly store fields in coupling.components.Component?
    coupled_component: Component  # component this field couples to
    timestep: Timestep  # generic timestep representation
    exchange_type: ExchangeType  # generic field exchange role
    metadata: str | None = None  # optional metadata


class FieldSet:
    """
    Set of fields.

    Can be used to collect fields and perform filtering operations for components, exchange types, etc.

    Example:
        fields = FieldSet()
        fields.add(Field(..., exchange_type=ExchangeType.PUT))
        fields.add(Field(..., exchange_type=ExchangeType.GET))
        put_fields = fields.filter(lambda f: f.exchange_type == ExchangeType.PUT)
    """

    def __init__(self, fields: set[Field] | None = None):
        """
        Initialize FieldSet.
        """
        self._fields = fields if fields is not None else set()

    def __iter__(self):
        return iter(self._fields)

    def __or__(self, other: "FieldSet") -> "FieldSet":
        assert isinstance(other, FieldSet), f"Can only merge FieldSet with FieldSet, got {type(other)}"
        return FieldSet(self._fields | other._fields)

    def __ior__(self, other: "FieldSet") -> "FieldSet":
        assert isinstance(other, FieldSet), f"Can only merge FieldSet with FieldSet, got {type(other)}"
        self._fields |= other._fields
        return self

    def is_empty(self) -> bool:
        return len(self._fields) == 0

    def all(self) -> set[Field]:
        return set(self._fields)

    def filter(self, condition: Callable[[Field], bool]) -> "FieldSet":
        return FieldSet(set(d for d in self._fields if condition(d)))

    def add(self, field: Field):
        assert field not in self._fields, f"Field {field} with name {field.name} already exists in FieldSet."
        self._fields.add(field)
