# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component, ExchangeKeySet

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, Timestep
from ebfm.core.config import ComponentId, TimeConfig
from ebfm.core.constants import DAYS_PER_YEAR


class ElmerIce(Component):
    """
    Component class for Elmer/Ice model coupling.
    """

    accepted_exchange_key_sets = (
        # All data is exchanged at once, i.e. the caller has to put and get everything in a single call.
        # The gradient fields dhdx and dhdy are not exchanged yet (their field definitions are commented out
        # below); add them to the get keys once they are enabled.
        ExchangeKeySet(
            name="exchange",
            put_keys={"T_ice", "smb", "runoff"},
            get_keys={"surface_elevation"},
        ),
    )

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
                    metadata="Surface mass balance",
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
                    name="surface_elevation",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface elevation (in m)",
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

    def _exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray],
        requested_key_set: ExchangeKeySet,
    ) -> dict[str, np.ndarray]:
        """
        Exchange data with Elmer/Ice.

        This component accepts a single key set, so everything is put and got here.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values Mapping of field names to fallback values to use if get fails
        @param[in] requested_key_set key set to be communicated, the only one this component accepts

        @returns dictionary of received field data. A requested field is not contained if it is not coupled, or
                 if no data was received for it and no fallback value was given.
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
        surface_elevation = self._get_if_coupled("surface_elevation", fallback_values=fallback_values)
        if surface_elevation is not None:
            received_data["surface_elevation"] = surface_elevation

        # The gradient fields have no field definition yet, so these two gets do nothing. Enabling their field
        # definitions also requires adding them to the get keys above, because the get keys a caller requests by
        # default are the coupled fields of this component.
        dhdx = self._get_if_coupled("dhdx", fallback_values=fallback_values)
        if dhdx is not None:
            received_data["dhdx"] = dhdx

        dhdy = self._get_if_coupled("dhdy", fallback_values=fallback_values)
        if dhdy is not None:
            received_data["dhdy"] = dhdy

        return received_data
