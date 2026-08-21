# SPDX-FileCopyrightText: 2025 EBFM Authors
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


class IconLand(Component):
    """
    Component class for coupling to the ICON land model (ICON-Land / JSBACH).

    The coupling to ICON-Land is separate from the coupling to the ICON atmosphere: ICON-Land
    is its own YAC component (``icon-land``) living on the ICON atmosphere processes.

    Currently EBFM only sends the ice fraction to ICON-Land. Fields received from ICON-Land
    will be added later.
    """

    def __init__(self, coupler: "Coupler", name: str = ComponentId.ICON_LAND.value):
        super().__init__(coupler, name)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to IconLand.
        """
        timestep = Timestep(value=time.time_step_iso8601())

        return FieldSet(
            {
                Field(
                    name="icefract",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Ice-covered fraction of the EBFM grid cell (1: glacier, 0: no glacier)",
                    exchange_type=ExchangeType.SOURCE,
                ),
            }
        )

    def exchange(
        self, data_to_exchange: Mapping[str, np.ndarray], fallback_values: Mapping[str, np.ndarray] = {}
    ) -> dict[str, np.ndarray]:
        """
        Exchange data with IconLand.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values optional Mapping of field names to fallback values to use if get fails

        @returns dictionary of received field data (currently empty)
        """
        received_data: dict[str, np.ndarray] = {}

        # Put data to IconLand
        self._put_if_coupled("icefract", data_to_exchange)

        # Get data from IconLand: nothing yet

        return received_data
