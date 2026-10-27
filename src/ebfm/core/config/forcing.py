# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Forcing configuration for EBFM.

This file provides the forcing configuration dataclass for EBFM.
"""

from enum import Enum
from pathlib import Path
from argparse import Namespace

from ebfm.core import logging

from .coupling import CouplingConfig

logger = logging.getLogger(__name__)


class ForcingType(Enum):
    """Enumeration of supported meteorological forcing types."""

    RANDOM = "random"
    CARRA2 = "carra2"
    ICON = "icon"


class ForcingConfig:
    """Configuration for meteorological forcing."""

    def __init__(self, coupling_config: CouplingConfig, args: Namespace):
        """
        Initialize forcing configuration from command line arguments.

        @param[in] coupling_config coupling configuration
        @param[in] args command line arguments
        """
        carra2_forcing_available = args.forcing_dir is not None
        icon_forcing_available = coupling_config.couple_to_icon_atmo

        assert not (
            carra2_forcing_available and icon_forcing_available
        ), "It is not allowed to use ICON coupling and CARRA2 forcing at the same time."

        if icon_forcing_available:
            self.forcing_type = ForcingType.ICON
            logger.info("Using ICON coupling for meteorological forcing.")
        elif carra2_forcing_available:
            self.forcing_type = ForcingType.CARRA2
            # Path to the folder containing NetCDF forcing files with meteorological data
            self.forcing_files_dir: Path = args.forcing_dir
            logger.info(f"Using CARRA2 forcing from {self.forcing_files_dir} for meteorological forcing.")
        else:
            self.forcing_type = ForcingType.RANDOM
            logger.info("Using random weather data for meteorological forcing.")
