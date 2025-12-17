# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict
import numpy as np

from elmer.mesh import Mesh as Grid  # for now use an alias

# from ebfm.geometry import Grid  # TODO: consider introducing a new data structure native to EBFM?
from ebfm.config import CouplingConfig

import logging
from ebfm.logger import deprecation

logger = logging.getLogger(__name__)


class Coupler:
    component_name: str  # name of this component in the coupler configuration

    _couple_to_elmer_ice: bool  # whether to couple with Elmer/Ice
    _couple_to_icon_atmo: bool  # whether to couple with ICON atmosphere

    @property
    def has_coupling(self) -> bool:
        return_value = self._couple_to_elmer_ice or self._couple_to_icon_atmo
        logger.debug(f"Component {self.component_name} has_coupling={return_value}")
        return return_value

    @property
    def couple_to_elmer_ice(self) -> bool:
        deprecation(
            logger, 'Coupler.couple_to_elmer_ice is deprecated, use Coupler.has_coupling_to("elmer_ice") instead.'
        )
        return self._couple_to_elmer_ice

    @property
    def couple_to_icon_atmo(self) -> bool:
        deprecation(
            logger, 'Coupler.couple_to_icon_atmo is deprecated, use Coupler.has_coupling_to("icon_atmo") instead.'
        )
        return self._couple_to_icon_atmo

    def has_coupling_to(self, component_name: str) -> bool:
        """
        Check if coupling to a specific component is enabled

        @param[in] component_name name of the component to check coupling for

        @returns True if coupling to the specified component is enabled, False otherwise
        """
        if component_name == "icon_atmo":
            return self._couple_to_icon_atmo
        elif component_name == "elmer_ice":
            return self._couple_to_elmer_ice
        else:
            raise ValueError(f"Unknown component name '{component_name}' for coupling check.")

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
        raise NotImplementedError("Coupler is an abstract base class and cannot be instantiated directly.")

    def setup(self, grid: Grid, time: Dict[str, float]):
        """
        Setup the coupling interface

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        raise NotImplementedError("setup method must be implemented in subclasses.")

    def add_grid(self, grid_name: str, grid: Grid):
        """
        Add grid to the Coupler interface
        """
        raise NotImplementedError("add_grid method must be implemented in subclasses.")

    def add_couples(self, time: Dict[str, float]):
        """
        Add coupling definitions to the Coupler interface
        """
        raise NotImplementedError("add_couples method must be implemented in subclasses.")

    def exchange_elmer_ice(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """Exchange data with Elmer Ice component

        @param[in] put_data dictionary of field names and their data to be exchanged with Elmer Ice component

        @returns dictionary of exchanged field data
        """
        deprecation(
            logger, 'Coupler.exchange_elmer_ice(data) is deprecated, use Coupler.exchange("elmer_ice", data) instead.'
        )
        return self.exchange("elmer_ice", data_to_exchange)

    def exchange_icon_atmo(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """Exchange data with ICON atmosphere component

        @param[in] put_data dictionary of field names and their data to be exchanged with ICON atmosphere component
        @returns dictionary of exchanged field data
        """
        deprecation(
            logger, 'Coupler.exchange_icon_atmo(data) is deprecated, use Coupler.exchange("icon_atmo", data) instead.'
        )
        return self.exchange("icon_atmo", data_to_exchange)

    def exchange(self, component_name: str, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """Exchange data with another component

        @param[in] component_name name of the component to exchange data with
        @param[in] data_to_exchange dictionary of field names and their data to be exchanged

        @returns dictionary of exchanged field data
        """
        raise NotImplementedError("Generic exchange method is not implemented. Use specific exchange methods.")

    def finalize(self):
        """
        Finalize the coupling interface
        """
        raise NotImplementedError("finalize method must be implemented in subclasses.")
