# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, days_to_iso


class ElmerIce(Component):
    """
    Component class for Elmer/Ice model coupling.
    """

    name = "elmer_ice"

    def __init__(self, coupler: "Coupler"):
        super().__init__(coupler)

    def get_field_definitions(self, time: Dict[str, float]) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to Elmer/Ice.
        """
        timestep = days_to_iso(time["dt"])

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

    def exchange(self, data_to_exchange: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Exchange data with Elmer/Ice.

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        received_data: Dict[str, np.ndarray] = {}

        # Put data to Elmer/Ice
        self._put_if_coupled("T_ice", data_to_exchange)
        self._put_if_coupled("smb", data_to_exchange)
        self._put_if_coupled("runoff", data_to_exchange)

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
