# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler
    from ebfm.coupling.fields.base import FieldSet


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

    def _put_if_coupled(self, field_name: str, data_to_exchange: Dict[str, np.ndarray]):
        """
        Put a source field if it is coupled.

        @param[in] field_name field name
        @param[in] data_to_exchange dictionary containing data to send
        """
        from ebfm.coupling.fields.base import ExchangeType

        if self._coupler.has_field(self.name, field_name, ExchangeType.SOURCE):
            assert (
                field_name in data_to_exchange
            ), f"Field '{field_name}' is missing in data_to_exchange for component '{self.name}'."
            self._coupler.put(self.name, field_name, data_to_exchange[field_name])

    def _get_if_coupled(self, field_name: str) -> Optional[np.ndarray]:
        """
        Get a target field from the coupler if it is coupled.

        @param[in] field_name field name

        @returns received field data if coupled, otherwise None
        """
        from ebfm.coupling.fields.base import ExchangeType

        if self._coupler.has_field(self.name, field_name, ExchangeType.TARGET):
            data, err = self._coupler.get(self.name, field_name)
            assert data is not None, f"Received data for field '{field_name}' is None. {err}"
            return data
        return None

    @abstractmethod
    def exchange(self, data_to_exchange: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Exchange of EBFM with this component

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        pass

    @abstractmethod
    def get_field_definitions(self, time: Dict[str, float]) -> "FieldSet":
        """
        Get field definitions for this component.
        Subclasses must implement this method.

        @param[in] time dictionary with time parameters
        @returns Set of Field objects for this component
        """
        pass
