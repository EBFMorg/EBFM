# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Time configuration for EBFM.

This file provides the time configuration dataclass for EBFM.
"""

from argparse import Namespace
from datetime import datetime, timedelta
from ..constants import SECONDS_PER_DAY

import logging

logger = logging.getLogger(__name__)


class TimeConfig:
    """
    Time configuration.
    """

    # Input time format for parsing command line arguments (e.g., "01-Jan-1979 00:00")
    input_time_format = "%d-%b-%Y %H:%M"
    # Used for showing time format in a human-readable way (e.g., in help messages)
    input_time_format_display = "DD-Mon-YYYY HH:MM"

    start_time: datetime  # Start time of the simulation (i.e., time at the beginning of the first time step)
    end_time: datetime  # End time of the simulation (i.e., time at the end of the last time step)
    time_step: timedelta  # Time step of the simulation
    dT_UTC: int  # Time difference relative to UTC in hours

    def __init__(self, args: Namespace):
        """
        Initialize time configuration from command line arguments.

        @param[in] args command line arguments
        """

        self.start_time = datetime.strptime(args.start_time, TimeConfig.input_time_format)
        self.end_time = datetime.strptime(args.end_time, TimeConfig.input_time_format)
        assert self.start_time < self.end_time, f"Start time {self.start_time} must be before end time {self.end_time}."

        assert args.time_step > 0, "Time step must be positive."
        self.time_step = timedelta(days=args.time_step)

        if self.time_step.total_seconds() > SECONDS_PER_DAY:
            logger.warning(
                f"Time step is {self.time_step.total_seconds()} seconds. Time steps larger than one day are not "
                f"recommended since this may lead to unexpected behavior or very long runtimes."
            )
        if SECONDS_PER_DAY % self.time_step.total_seconds() != 0:
            logger.warning(
                f"Time step of {self.time_step.total_seconds()} seconds does not evenly divide one "
                f"day ({SECONDS_PER_DAY} seconds). This may lead to unexpected behavior."
            )

        self.dT_UTC = 1  # Time difference relative to UTC in hours (hard-coded for now)

    def tn(self) -> int:
        """Calculate the number of time steps.

        @returns Number of time steps
        """
        total_seconds = (self.end_time - self.start_time).total_seconds()
        step_seconds = self.time_step.total_seconds()
        assert total_seconds % step_seconds == 0, "Time interval must be divisible by time step."
        return int(round(total_seconds / step_seconds))

    def time_step_in_days(self) -> float:
        """Get the time step size in days.

        @returns Time step size in days
        """
        return self.time_step.total_seconds() / SECONDS_PER_DAY

    def time_step_iso8601(self) -> str:
        """Get the time step size in ISO 8601 duration format (e.g., "P0DT3H0M0S" for a 3-hour time step).

        @returns Time step size in ISO 8601 duration format
        """
        import pandas as pd

        dt = pd.Timedelta(days=self.time_step_in_days())
        return dt.isoformat()

    def to_dict(self) -> dict:
        """Convert time configuration to a dictionary.

        @returns Dictionary representation of the time configuration
        """
        return {
            "ts": self.start_time,
            "te": self.end_time,
            "dt": self.time_step_in_days(),
            "tn": self.tn(),
            "dT_UTC": self.dT_UTC,
        }
