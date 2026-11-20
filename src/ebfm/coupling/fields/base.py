# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from ebfm.coupling.components.base import Component
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
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

    A field is identified by its name and the component it couples to: two fields that agree in these are equal,
    whatever their other attributes.
    """

    name: str  # name of the field
    # TODO: remove coupler_component and directly store fields in coupling.components.Component?
    coupled_component: Component  # component this field couples to
    timestep: Timestep = field(compare=False)  # generic timestep representation
    exchange_type: ExchangeType = field(compare=False)  # generic field exchange role
    metadata: str | None = field(default=None, compare=False)  # optional metadata
    # Plausible (min, max) of received values, in the units of the field. Coupler.get rejects values outside.
    valid_range: tuple[float, float] | None = field(default=None, compare=False)


class FieldSet:
    """
    Set of fields.

    Holds at most one field per name and component (see Field); adding a second one is rejected.

    Can be used to collect fields and perform filtering operations for components, exchange types, etc.

    Example:
        fields = FieldSet()
        fields.add(Field(..., exchange_type=ExchangeType.PUT))
        fields.add(Field(..., exchange_type=ExchangeType.GET))
        put_fields = fields.filter(lambda f: f.exchange_type == ExchangeType.PUT)
    """

    def __init__(self, fields: Iterable[Field] = ()):
        """
        Initialize FieldSet.

        @param[in] fields fields to add, see add
        """
        self._fields: set[Field] = set()
        for f in fields:
            self.add(f)

    def __iter__(self):
        return iter(self._fields)

    def __or__(self, other: "FieldSet") -> "FieldSet":
        assert isinstance(other, FieldSet), f"Can only merge FieldSet with FieldSet, got {type(other)}"
        return FieldSet([*self._fields, *other._fields])

    def __ior__(self, other: "FieldSet") -> "FieldSet":
        assert isinstance(other, FieldSet), f"Can only merge FieldSet with FieldSet, got {type(other)}"
        for f in other:
            self.add(f)
        return self

    def is_empty(self) -> bool:
        return len(self._fields) == 0

    def all(self) -> set[Field]:
        return set(self._fields)

    def filter(self, condition: Callable[[Field], bool]) -> "FieldSet":
        return FieldSet(d for d in self._fields if condition(d))

    def get(self, field_name: str, component_name: str | None = None) -> Field:
        """
        Get the field of the given name.

        @param[in] field_name name of the field
        @param[in] component_name name of the component the field couples to; needed if the set holds the fields of
                   several components

        @returns the field

        @raises KeyError if not exactly one field matches
        """
        fields = self.filter(
            lambda f: f.name == field_name and (component_name is None or f.coupled_component.name == component_name)
        ).all()
        if len(fields) != 1:
            raise KeyError(
                f"Expected exactly one field named '{field_name}'"
                + ("" if component_name is None else f" for component '{component_name}'")
                + f", found {len(fields)}."
            )
        (field,) = fields
        return field

    def add(self, field: Field):
        assert (
            field not in self._fields
        ), f"Field '{field.name}' of component '{field.coupled_component.name}' is defined more than once."
        self._fields.add(field)
