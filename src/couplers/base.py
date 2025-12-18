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
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Component(Enum):
    """
    Other components which this model can couple to
    """

    elmer_ice = auto()
    icon_atmo = auto()


class Coupler:
    """
    Abstract base class for couplers. Implements the strategy pattern to support different coupling libraries.
    """

    component_name: str  # name of this component in the coupler configuration

    _couples_to: Dict[Component, bool] = {}  # dict gives information whether to couple with given component

    @property
    def has_coupling(self) -> bool:
        """
        Check if coupling to any component is enabled

        @returns True if coupling to any component is enabled, False otherwise
        """
        deprecation(
            logger,
            "Coupler.has_coupling is deprecated, use Coupler.has_coupling_to(comp_name) if you want to check for "
            "specific components.\n"
            "### Design recommendations ###\n"
            "It is recommended to follow the strategy pattern and initialize your Coupler as a DummyCoupler, "
            "YACCoupler or similar to avoid general has_coupling checks since this usually leads to a better design.\n"
            "Alternatively, you can check for the type of your coupler. A DummyCoupler never has a coupling and you "
            "can use this to perform checks similar to has_coupling.\n"
            "##############################",
        )
        return_value = any(self._couples_to.values())
        logger.debug(f"Component {self.component_name} has_coupling={return_value}")
        return return_value

    @property
    def couple_to_elmer_ice(self) -> bool:
        deprecation(
            logger, 'Coupler.couple_to_elmer_ice is deprecated, use Coupler.has_coupling_to("elmer_ice") instead.'
        )
        return self.has_coupling_to("elmer_ice")

    @property
    def couple_to_icon_atmo(self) -> bool:
        deprecation(
            logger, 'Coupler.couple_to_icon_atmo is deprecated, use Coupler.has_coupling_to("icon_atmo") instead.'
        )
        return self.has_coupling_to("icon_atmo")

    def has_coupling_to(self, component_name: str) -> bool:
        """
        Check if coupling to a specific component is enabled

        @param[in] component_name name of the component to check coupling for

        @returns True if coupling to the specified component is enabled, False otherwise
        """
        component = Component[component_name]
        return self._couples_to[component]

    def __init__(self, coupling_config: CouplingConfig):
        """
        Create Coupler object

        @param[in] coupling_config configuration of the coupling
        """
        raise NotImplementedError("Coupler is an abstract base class and cannot be instantiated directly.")

    def setup(self, grid: Grid | Dict, time: Dict[str, float]):
        """
        Setup the coupling interface

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling fields are defined
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        if isinstance(grid, dict):
            deprecation(
                logger,
                "Providing grid as Dict is not recommended. Please provide a grid of type coupler.Grid. "
                'Usually you can get this Grid by accessing grid["mesh"] from the grid dictionary.',
            )
            grid = grid["mesh"]

        self._setup(grid, time)

    def _setup(self, grid: Grid, time: Dict[str, float]):
        """
        Implementation of Coupler.setup

        @param[in] grid Grid used by EBFM where coupling fields are defined
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        raise NotImplementedError("setup method must be implemented in subclasses.")

    def _add_grid(self, grid_name: str, grid: Grid):
        """
        Add grid to the Coupler interface
        """
        raise NotImplementedError("add_grid method must be implemented in subclasses.")

    def _add_couples(self, time: Dict[str, float]):
        """
        Add coupling definitions to the Coupler interface
        """
        raise NotImplementedError("add_couples method must be implemented in subclasses.")

    def exchange_elmer_ice(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with Elmer Ice component

        @param[in] put_data dictionary of field names and their data to be exchanged with Elmer Ice component

        @returns dictionary of exchanged field data
        """
        deprecation(
            logger, 'Coupler.exchange_elmer_ice(data) is deprecated, use Coupler.exchange("elmer_ice", data) instead.'
        )
        return self.exchange(Component.elmer_ice.name, data_to_exchange)

    def exchange_icon_atmo(self, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with ICON atmosphere component

        @param[in] put_data dictionary of field names and their data to be exchanged with ICON atmosphere component
        @returns dictionary of exchanged field data
        """
        deprecation(
            logger, 'Coupler.exchange_icon_atmo(data) is deprecated, use Coupler.exchange("icon_atmo", data) instead.'
        )
        return self.exchange(Component.icon_atmo.name, data_to_exchange)

    def exchange(self, component_name: str, data_to_exchange: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Exchange data with another component

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
