# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from couplers import Coupler
from couplers.base import Grid, Dict, CouplingConfig, Component

import logging

logger = logging.getLogger(__name__)


class DummyCoupler(Coupler):
    """
    A dummy coupler implementation that does nothing.
    This can be used when no coupling is required.
    """

    # DummyCoupler couples to none of the available components
    _couples_to: Dict[Component, bool] = {c: False for c in Component}

    def __init__(self, coupling_config: CouplingConfig):
        self.component_name = coupling_config.component_name
        logger.debug(f"DummyCoupler created for component '{self.component_name}'.")

    def _setup(self, grid: Grid, time: Dict[str, float]):
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
