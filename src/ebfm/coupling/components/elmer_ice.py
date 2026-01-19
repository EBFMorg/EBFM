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


class ElmerIce(Component):
    """
    Component class for Elmer/Ice model coupling.
    """

    name = "elmer_ice"

    def __init__(self, coupler: "Coupler"):
        super().__init__(coupler)

    def _yac_field_definitions(self, time: Dict[str, float]) -> Set[Field]:
        """
        Get field definitions for EBFM coupling to Elmer/Ice using YAC coupler.
        """
        assert coupling_supported, "Coupling support is required for YAC fields."

        timestep_value = days_to_iso(time["dt"])
        timestep = Timestep(value=timestep_value, format=yac.TimeUnit.ISO_FORMAT)

        return {
            YACField(
                name="T_ice",
                coupled_component=self,
                timestep=timestep,
                metadata="Near surface temperature at Ice surface (in K)",
                exchange_type=yac.ExchangeType.SOURCE,
            ),
            YACField(
                name="smb",
                coupled_component=self,
                timestep=timestep,
                exchange_type=yac.ExchangeType.SOURCE,
            ),
            YACField(
                name="runoff",
                coupled_component=self,
                timestep=timestep,
                metadata="Runoff",
                exchange_type=yac.ExchangeType.SOURCE,
            ),
            YACField(
                name="h",
                coupled_component=self,
                timestep=timestep,
                metadata="Surface height (in m)",
                exchange_type=yac.ExchangeType.TARGET,
            ),
            # YACField(
            #     name="dhdx",
            #     coupled_component=self,
            #     timestep=timestep,
            #     metadata="Surface slope in x direction",
            #     exchange_type=yac.ExchangeType.TARGET,
            # ),
            # YACField(
            #     name="dhdy",
            #     coupled_component=self,
            #     timestep=timestep,
            #     metadata="Surface slope in y direction",
            #     exchange_type=yac.ExchangeType.TARGET,
            # ),
        }

    def _yac_exchange(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange of EBFM with Elmer/Ice using YAC coupler.

        @param[in] data_to_exchange dictionary of field names and their data to be sent

        @returns dictionary of received field data
        """
        assert coupling_supported, "Coupling support is required for YAC exchange."

        received_data: Dict[str, np.array] = {}

        # Put data to Elmer/Ice
        self._coupler.put(self.name, "T_ice", data_to_exchange["T_ice"])
        self._coupler.put(self.name, "smb", data_to_exchange["smb"])
        self._coupler.put(self.name, "runoff", data_to_exchange["runoff"])

        # Get data from Elmer/Ice
        received_data["h"] = self._coupler.get(self.name, "h")
        # received_data["dhdx"] = self._coupler.get(self.name, "dhdx")
        # received_data["dhdy"] = self._coupler.get(self.name, "dhdy")

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
