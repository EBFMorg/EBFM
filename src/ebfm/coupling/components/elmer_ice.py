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
from ebfm.core.constants import DAYS_PER_YEAR


class ElmerIce(Component):
    """
    Component class for Elmer/Ice model coupling.
    """

    def __init__(self, coupler: "Coupler", name: str = ComponentId.ELMER_ICE.value):
        super().__init__(coupler, name)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to Elmer/Ice.
        """
        timestep = Timestep(value=time.time_step_iso8601())

        return FieldSet(
            {
                Field(
                    name="T_ice",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Near surface temperature at Ice surface (in K)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="smb",
                    coupled_component=self,
                    timestep=timestep,
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="runoff",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Runoff",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="h",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface height (in m)",
                    exchange_type=ExchangeType.TARGET,
                ),
                # Field(
                #     name="dhdx",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Surface slope in x direction",
                #     exchange_type=ExchangeType.TARGET,
                # ),
                # Field(
                #     name="dhdy",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Surface slope in y direction",
                #     exchange_type=ExchangeType.TARGET,
                # ),
            }
        )

    def exchange(self, data_to_exchange: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Exchange data with Elmer/Ice.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent

        @returns dictionary of received field data
        """
        received_data: dict[str, np.ndarray] = {}

        # For fields representing rates (e.g. SMB, runoff), we need to convert them from per timestep to per year
        # before sending to Elmer/Ice, which expects annual values.
        def map_per_timestep_to_per_year(x_per_timestep: np.ndarray) -> np.ndarray:
            x_per_day = x_per_timestep / self._coupler.get_time_step_in_days()
            x_per_year = x_per_day * DAYS_PER_YEAR
            return x_per_year

        # Put data to Elmer/Ice
        self._put_if_coupled("T_ice", data_to_exchange)
        self._put_if_coupled("smb", data_to_exchange, transform=map_per_timestep_to_per_year)
        self._put_if_coupled("runoff", data_to_exchange, transform=map_per_timestep_to_per_year)

        # Get data from Elmer/Ice
        h = self._get_if_coupled("h")
        if h is not None:
            received_data["h"] = h

        dhdx = self._get_if_coupled("dhdx")
        if dhdx is not None:
            received_data["dhdx"] = dhdx

        dhdy = self._get_if_coupled("dhdy")
        if dhdy is not None:
            received_data["dhdy"] = dhdy

        return received_data
