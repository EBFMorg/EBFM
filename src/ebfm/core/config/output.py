# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Output configuration for EBFM.

This file provides the configuration of the NetCDF output file for EBFM.
"""

from argparse import Namespace
from datetime import timedelta
from pathlib import Path


class OutputConfig:
    """Configuration of the NetCDF output file."""

    def __init__(self, args: Namespace):
        """
        Initialize output configuration from command line arguments.

        @param[in] args command line arguments
        """
        # Relative paths are relative to the working directory
        self.output_file: Path = Path.cwd() / args.output_file
        # Interval between two output records
        self.output_frequency: timedelta = args.output_frequency
