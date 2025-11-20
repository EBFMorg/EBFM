# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Dict

from elmer.mesh import Mesh as Grid  # for now use an alias


import logging

logger = logging.getLogger(__name__)


class Coupler:
    component_name: str  # name of this component in the coupler configuration

    couple_to_elmer_ice: bool  # whether to couple with Elmer/Ice
    couple_to_icon_atmo: bool  # whether to couple with ICON atmosphere

    @property
    def has_coupling(self) -> bool:
        return_value = self.couple_to_elmer_ice or self.couple_to_icon_atmo
        logger.debug(f"Component {self.component_name} has_coupling={return_value}")
        return return_value

    def __init__(self, component_name: str, coupler_config: Path = None):
        """
        Create Coupler object

        @param[in] component_name name of this component in the coupler configuration
        @param[in] coupler_config path to the coupler configuration file
        """
        raise NotImplementedError("Coupler is an abstract base class and cannot be instantiated directly.")

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


class NoCoupler(Coupler):
    couple_to_icon_atmo: bool = False
    couple_to_elmer_ice: bool = False

    def __init__(self, component_name: str):
        self.component_name = component_name
        logger.debug(f"NoCoupler created for component '{component_name}'.")
