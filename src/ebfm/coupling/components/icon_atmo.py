# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component

from ebfm.coupling.fields import FieldSet
from ebfm.coupling.couplers.helpers import coupling_supported

if coupling_supported:
    # TODO: Try to remove YAC specific imports from here
    import yac
    from ebfm.coupling.fields.yacField import YACField, Timestep, days_to_iso


class IconAtmo(Component):
    """
    Component class for ICON atmosphere model coupling.
    """

    name = "icon_atmo"

    def __init__(self, coupler: "Coupler"):
        super().__init__(coupler)

    def _yac_field_definitions(self, time: Dict[str, float]) -> FieldSet:
        """
        Get field definitions for EBFM coupling to IconAtmo using YAC coupler.
        """
        assert coupling_supported, "Coupling support is required for YAC fields."

        timestep_value = days_to_iso(time["dt"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        return FieldSet(
            {
                # YACField(
                #     name="albedo",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Albedo of the ice surface",
                #     exchange_type=yac.ExchangeType.SOURCE,
                # ),
                YACField(
                    name="pr",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Precipitation rate (in kg m-2 s-1)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="pr_snow",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Precipitation rate of snow (in kg m-2 s-1)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="rsds",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Downward shortwave radiation flux (in W m-2)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="rlds",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Downward longwave radiation flux (in W m-2)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="sfcwind",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Wind speed at surface (in m s-1)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="clt",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Cloud cover (in fraction)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="tas",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Temperature at surface (in K)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="huss",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Specific humidity at surface (in kg kg-1)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
                YACField(
                    name="sfcpres",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface pressure (in Pa)",
                    exchange_type=yac.ExchangeType.TARGET,
                ),
            }
        )

    def _yac_exchange(self, data_to_exchange: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Exchange of EBFM with IconAtmo using YAC coupler.

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns tuple of (received field data, error codes). An error code of None indicates
                 successful exchange for that field.
        """
        assert coupling_supported, "Coupling support is required for YAC exchange."

        received_data: Dict[str, np.ndarray] = {}

        # Put data to IconAtmo
        if self._coupler.has_field(self.name, "albedo", yac.ExchangeType.SOURCE):
            self._coupler.put(self.name, "albedo", data_to_exchange["albedo"])

        # Get data from IconAtmo
        if self._coupler.has_field(self.name, "pr", yac.ExchangeType.TARGET):
            pr, err = self._coupler.get(self.name, "pr")
            assert pr is not None, f"Received data for field 'pr' is None. {err}"
            received_data["pr"] = pr

        if self._coupler.has_field(self.name, "pr_snow", yac.ExchangeType.TARGET):
            pr_snow, err = self._coupler.get(self.name, "pr_snow")
            assert pr_snow is not None, f"Received data for field 'pr_snow' is None. {err}"
            received_data["pr_snow"] = pr_snow

        if self._coupler.has_field(self.name, "rsds", yac.ExchangeType.TARGET):
            rsds, err = self._coupler.get(self.name, "rsds")
            assert rsds is not None, f"Received data for field 'rsds' is None. {err}"
            received_data["rsds"] = rsds

        if self._coupler.has_field(self.name, "rlds", yac.ExchangeType.TARGET):
            rlds, err = self._coupler.get(self.name, "rlds")
            assert rlds is not None, f"Received data for field 'rlds' is None. {err}"
            received_data["rlds"] = rlds

        if self._coupler.has_field(self.name, "sfcwind", yac.ExchangeType.TARGET):
            sfcwind, err = self._coupler.get(self.name, "sfcwind")
            assert sfcwind is not None, f"Received data for field 'sfcwind' is None. {err}"
            received_data["sfcwind"] = sfcwind

        if self._coupler.has_field(self.name, "clt", yac.ExchangeType.TARGET):
            clt, err = self._coupler.get(self.name, "clt")
            assert clt is not None, f"Received data for field 'clt' is None. {err}"
            received_data["clt"] = clt

        if self._coupler.has_field(self.name, "tas", yac.ExchangeType.TARGET):
            tas, err = self._coupler.get(self.name, "tas")
            assert tas is not None, f"Received data for field 'tas' is None. {err}"
            received_data["tas"] = tas

        if self._coupler.has_field(self.name, "huss", yac.ExchangeType.TARGET):
            huss, err = self._coupler.get(self.name, "huss")
            assert huss is not None, f"Received data for field 'huss' is None. {err}"
            received_data["huss"] = huss

        if self._coupler.has_field(self.name, "sfcpres", yac.ExchangeType.TARGET):
            sfcpres, err = self._coupler.get(self.name, "sfcpres")
            assert sfcpres is not None, f"Received data for field 'sfcpres' is None. {err}"
            received_data["sfcpres"] = sfcpres

        return received_data

    def get_field_definitions(self, time: Dict[str, float]) -> FieldSet:
        """
        Get field definitions for EBFM coupling.

        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """

        if self._uses_coupler("YACCoupler"):
            return self._yac_field_definitions(time)
        else:
            raise NotImplementedError(
                f"The component {self.name} was configured with the unsupported coupler {type(self._coupler)}."
                f"Note: {type(self)} only supports YACCoupler at the moment. "
            )

    def exchange(self, data_to_exchange: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Exchange data with IconAtmo.

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        if self._uses_coupler("YACCoupler"):
            return self._yac_exchange(data_to_exchange)
        else:
            raise NotImplementedError(
                f"The component {self.name} was configured with the unsupported coupler {type(self._coupler)}."
                f"Note: {type(self)} only supports YACCoupler at the moment. "
            )
