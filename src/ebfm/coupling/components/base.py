# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Mapping, Callable
import numpy as np

from ebfm.core import logging

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler
    from ebfm.coupling.fields.base import FieldSet


def identity(x: np.ndarray) -> np.ndarray:
    """
    Identity function for default transform.

    @param[in] x input data

    @returns the input data unchanged
    """
    return x


class Component(ABC):
    """
    Abstract base class for coupling components.
    Each component owns its fields as an instance attribute.
    """

    def __init__(self, coupler: "Coupler", name: str):
        """
        Initialize the component.

        @param[in] coupler Coupler instance to use for coupling
        @param[in] name unique name of the component
        """
        self._coupler = coupler
        self.name = name
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

    def _put_if_coupled(
        self, field_name: str, data_to_exchange: Mapping[str, np.ndarray], transform: Callable = identity
    ):
        """
        Put a source field if it is coupled.

        @param[in] field_name field name
        @param[in] data_to_exchange dictionary containing data to send
        @param[in] transform optional function to apply to the data before sending (e.g. for unit conversion)
        """
        from ebfm.coupling.fields.base import ExchangeType

        if self._coupler.has_field(self.name, field_name, ExchangeType.SOURCE):
            assert (
                field_name in data_to_exchange
            ), f"Field '{field_name}' is missing in data_to_exchange for component '{self.name}'."
            logger.debug(f"Putting data for field '{field_name}' to coupler: {data_to_exchange[field_name]}")
            err = self._coupler.put(self.name, field_name, transform(data_to_exchange[field_name]))
            if err:
                logger.warning(
                    f"Put for {field_name=} returned unexpected exit code ({err=}). "
                    "Please report this to the developers."
                )

    def _get_if_coupled(
        self, field_name: str, transform: Callable = identity, fallback_values: Mapping[str, np.ndarray] = {}
    ) -> np.ndarray | None:
        """
        Get a target field from the coupler if it is coupled.

        @param[in] field_name field name
        @param[in] transform optional function to apply to the received data (e.g. for unit conversion)
        @param[in] fallback_values optional dictionary of fallback values to use if get fails

        @returns received field data if coupled, otherwise None
        """
        from ebfm.coupling.fields.base import ExchangeType

        if not self._coupler.has_field(self.name, field_name, ExchangeType.TARGET):
            logger.debug(f"Field '{field_name}' is not coupled for component '{self.name}', skipping get.")
            return None

        data, err = self._coupler.get(self.name, field_name)

        if err:
            logger.warning(f"Get for {field_name=} returned exit code ({err=}).")
            if fallback := fallback_values.get(field_name) is not None:
                logger.warning(
                    f"Using fallback value for '{field_name}' as no data was received for this field from {self.name}."
                )
                return fallback
            return None

        assert data is not None, f"Received data for field '{field_name}' is None. {err}"
        logger.debug(f"Received data for field '{field_name}' from coupler: {data}")
        return transform(data)

    @abstractmethod
    def exchange(
        self, data_to_exchange: Mapping[str, np.ndarray], fallback_values: Mapping[str, np.ndarray] = {}
    ) -> dict[str, np.ndarray]:
        """
        Exchange of EBFM with this component

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values optional Mapping of field names to fallback values to use if get fails

        @returns dictionary of received field data
        """
        pass

    @abstractmethod
    def get_field_definitions(self, time: dict[str, float]) -> "FieldSet":
        """
        Get field definitions for this component.
        Subclasses must implement this method.

        @param[in] time dictionary with time parameters
        @returns Set of Field objects for this component
        """
        pass
