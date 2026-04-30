# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, Timestep
from ebfm.core.config import ComponentId, TimeConfig

from ebfm.core import logging

logger = logging.getLogger(__name__)


class Dummy(Component):
    """
    A dummy component for demonstration and testing purposes.
    """

    def __init__(self, coupler: "Coupler", name: str = ComponentId.DUMMY.value):
        super().__init__(coupler, name)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        """
        Get field definitions for the dummy component.

        We only read data from that component.
        """
        timestep = Timestep(value=time.time_step_iso8601())

        return FieldSet(
            {
                Field(
                    name="dummy_field",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="A dummy field for testing purposes",
                    exchange_type=ExchangeType.TARGET,
                )
            }
        )

    def exchange(self, data_to_exchange: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Exchange data with the dummy component.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent

        @returns dictionary of received field data
        """
        received_data: dict[str, np.ndarray] = {}

        field_name = "dummy_field"
        dummy_data = self._get_if_coupled(field_name)
        if dummy_data is not None:
            logger.debug(f"Received the following data for {field_name=}: {dummy_data}")
            received_data[field_name] = dummy_data
        else:
            logger.debug(f"No data received for {field_name=} (not coupled or not sent by the component).")

        return received_data
