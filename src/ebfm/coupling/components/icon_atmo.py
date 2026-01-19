# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from coupling.couplers.base import Coupler

from coupling.components.base import Component

from coupling.fields import Field
from coupling.couplers.helpers import coupling_supported

if coupling_supported:
    # TODO: Try to remove YAC specific imports from here
    import yac
    from coupling.fields.yacField import YACField, Timestep, days_to_iso


class IconAtmo(Component):
    """
    Component class for ICON atmosphere model coupling.
    """

    name = "icon_atmo"

    def __init__(self, coupler: "Coupler"):
        super().__init__(coupler)

    def _yac_field_definitions(self, time: Dict[str, float]) -> Set[Field]:
        """
        Get field definitions for EBFM coupling to IconAtmo using YAC coupler.
        """
        assert coupling_supported, "Coupling support is required for YAC fields."

        timestep_value = days_to_iso(time["dt"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        return {
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
            # YACField(
            #     name="huss",
            #     component=self,
            #     timestep=timestep,
            #     metadata="Specific humidity at surface (in kg kg-1)"
            #     exchange_type=yac.ExchangeType.TARGET,
            # ),
            # YACField(
            #     name="sfcPressure",
            #     component=self,
            #     timestep=timestep,
            #     metadata="Surface pressure (in Pa)"
            #     exchange_type=yac.ExchangeType.TARGET,
            # ),
        }

    def _yac_exchange(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange of EBFM with IconAtmo using YAC coupler.

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        assert coupling_supported, "Coupling support is required for YAC exchange."

        received_data: Dict[str, np.array] = {}

        # Put data to IconAtmo
        # self._coupler.put(self.name, "albedo", data_to_exchange["albedo"])

        # Get data from IconAtmo
        received_data["pr"] = self._coupler.get(self.name, "pr")
        received_data["pr_snow"] = self._coupler.get(self.name, "pr_snow")
        received_data["rsds"] = self._coupler.get(self.name, "rsds")
        received_data["rlds"] = self._coupler.get(self.name, "rlds")
        received_data["sfcwind"] = self._coupler.get(self.name, "sfcwind")
        received_data["clt"] = self._coupler.get(self.name, "clt")
        received_data["tas"] = self._coupler.get(self.name, "tas")
        # received_data["huss"] = self._coupler.get(self.name,"huss")
        # received_data["sfcPressure"] = self._coupler.get(self.name,"sfcPressure")

        return received_data

    def get_field_definitions(self, time: Dict[str, float]) -> Set[Field]:
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

    def exchange(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        if self._uses_coupler("YACCoupler"):
            return self._yac_exchange(data_to_exchange)
        else:
            raise NotImplementedError(
                f"The component {self.name} was configured with the unsupported coupler {type(self._coupler)}."
                f"Note: {type(self)} only supports YACCoupler at the moment. "
            )
