# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from couplers import Coupler
from couplers.base import Grid, Dict, CouplingConfig

import logging

logger = logging.getLogger(__name__)


class DummyCoupler(Coupler):
    """
    A dummy coupler implementation that does nothing.
    This can be used when no coupling is required.
    """

    couple_to_icon_atmo: bool = False
    couple_to_elmer_ice: bool = False

    def __init__(self, coupling_config: CouplingConfig):
        self.component_name = coupling_config.component_name
        logger.debug(f"DummyCoupler created for component '{self.component_name}'.")

    def setup(self, grid: Grid, time: Dict[str, float]):
        """Setup the coupling interface (does nothing for DummyCoupler)

        Performs initialization operations after init and before entering the
        time loop

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        logger.debug("Setup coupling...")
        logger.debug("Do nothing for DummyCoupler.")

    def finalize(self):
        """Finalize the coupling interface (does nothing for DummyCoupler)"""
        logger.debug("Finalizing coupling...")
        logger.debug("No coupling to finalize for DummyCoupler.")
