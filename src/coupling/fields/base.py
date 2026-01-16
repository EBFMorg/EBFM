# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from coupling.components.base import Component
from dataclasses import dataclass
from typing import Set, Callable


@dataclass(frozen=True)
class Field:
    """
    Object for definition of a generic field.
    """

    name: str  # name of the field
    # TODO: remove coupler_component and directly store fields in coupling.components.Component?
    coupled_component: Component  # component this field couples to


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
