# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Forcing configuration for EBFM.

This file provides the forcing configuration dataclass for EBFM.
"""

from enum import Enum
from argparse import Namespace

from ebfm.core import logging

from .coupling import CouplingConfig

logger = logging.getLogger(__name__)


class ForcingType(Enum):
    """Enumeration of supported meteorological forcing types."""

    RANDOM = "random"
    ICON = "icon"


class ForcingConfig:
    """Configuration for meteorological forcing."""

    def __init__(self, coupling_config: CouplingConfig, args: Namespace):
        """
        Initialize forcing configuration from command line arguments.

        @param[in] coupling_config coupling configuration
        @param[in] args command line arguments
        """
        icon_forcing_available = coupling_config.couple_to_icon_atmo

        if icon_forcing_available:
            self.forcing_type = ForcingType.ICON
            logger.info("Using ICON coupling for meteorological forcing.")
        else:
            self.forcing_type = ForcingType.RANDOM
            logger.info("Using random weather data for meteorological forcing.")
