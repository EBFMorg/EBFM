# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Set, Dict, TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from coupling.couplers.base import Coupler
    from coupling.fields.yacField import Field


class Component(ABC):
    """
    Abstract base class for coupling components.
    Each component owns its fields as an instance attribute.
    """

    name: str  # name of this component

    def __init__(self, coupler: "Coupler"):
        self._coupler = coupler
        pass

    def _uses_coupler(self, coupler_class_type) -> bool:
        """
        Check if the coupler is of a specific class type.

        This function is provided to avoid importing coupling libraries in component modules which would result in a
        circular dependency. You can check by providing the class name of the respective coupler

        Example: self._uses_coupler("YACCoupler")

        @param[in] coupler_class_type name of class type to check against

        @returns True if the coupler is of the specified class type, False otherwise
        """
        return self._coupler.__class__.__name__ == coupler_class_type

    @abstractmethod
    def exchange(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange of EBFM with this component

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        pass

    @abstractmethod
    def get_field_definitions(self, time: Dict[str, float]) -> Set["Field"]:
        """
        Get field definitions for this component.
        Subclasses must implement this method.

        @param[in] time dictionary with time parameters
        @returns Set of Field objects for this component
        """
        pass
