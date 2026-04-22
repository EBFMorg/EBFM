# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np
from ebfm.core.constants import SECONDS_PER_DAY

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, Timestep
from ebfm.core.config import TimeConfig


class IconAtmo(Component):
    """
    Component class for ICON atmosphere model coupling.
    """

    name = "icon_atmo"

    def __init__(self, coupler: "Coupler"):
        super().__init__(coupler)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to IconAtmo.
        """
        timestep = Timestep(value=time.time_step_iso8601())

        return FieldSet(
            {
                # Field(
                #     name="albedo",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Albedo of the ice surface",
                #     exchange_type=ExchangeType.SOURCE,
                # ),
                Field(
                    name="pr",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Precipitation rate (in kg m-2 s-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="pr_snow",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Precipitation rate of snow (in kg m-2 s-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="rsds",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Downward shortwave radiation flux (in W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="rlds",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Downward longwave radiation flux (in W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="sfcwind",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Wind speed at surface (in m s-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="clt",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Cloud cover (in fraction)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="tas",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Temperature at surface (in K)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="huss",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Specific humidity at surface (in kg kg-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="sfcpres",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface pressure (in Pa)",
                    exchange_type=ExchangeType.TARGET,
                ),
            }
        )

    def exchange(self, data_to_exchange: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Exchange data with IconAtmo.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent

        @returns dictionary of received field data
        """
        received_data: dict[str, np.ndarray] = {}

        # We need to convert precipitation received from ICON from kg / m^2 / s
        # to m w.e. (per EBFM timestep)
        def map_pr_to_ebfm(precipitation: np.ndarray) -> np.ndarray:
            mwe_per_second = precipitation * 1e-3
            mwe_per_day = mwe_per_second * SECONDS_PER_DAY
            mwe_per_timestep = mwe_per_day * self._coupler.get_time_step_in_days()
            return mwe_per_timestep

        # Put data to IconAtmo
        self._put_if_coupled("albedo", data_to_exchange)

        # Get data from IconAtmo
        pr = self._get_if_coupled("pr", transform=map_pr_to_ebfm)
        if pr is not None:
            received_data["pr"] = pr

        # TODO: check what units are needed for pr_snow in EBFM
        pr_snow = self._get_if_coupled("pr_snow")
        if pr_snow is not None:
            received_data["pr_snow"] = pr_snow

        rsds = self._get_if_coupled("rsds")
        if rsds is not None:
            received_data["rsds"] = rsds

        rlds = self._get_if_coupled("rlds")
        if rlds is not None:
            received_data["rlds"] = rlds

        sfcwind = self._get_if_coupled("sfcwind")
        if sfcwind is not None:
            received_data["sfcwind"] = sfcwind

        clt = self._get_if_coupled("clt")
        if clt is not None:
            received_data["clt"] = clt

        tas = self._get_if_coupled("tas")
        if tas is not None:
            received_data["tas"] = tas

        huss = self._get_if_coupled("huss")
        if huss is not None:
            received_data["huss"] = huss

        sfcpres = self._get_if_coupled("sfcpres")
        if sfcpres is not None:
            received_data["sfcpres"] = sfcpres

        return received_data
