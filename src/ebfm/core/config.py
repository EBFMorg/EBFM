# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for EBFM.

This file exposes a some configuration dataclasses for EBFM components.
"""

from argparse import Namespace
from pathlib import Path
from enum import Enum
from typing import Optional

from .grid import GridInputType

from datetime import datetime, timedelta
from .constants import SECONDS_PER_DAY

import logging

logger = logging.getLogger(__name__)


class FieldValidationLevel(Enum):
    """Level of validation for field exchange type checks."""

    FATAL = "FATAL"  # Raise an exception on mismatch
    WARNING = "WARNING"  # Log a warning on mismatch
    SILENT = "SILENT"  # Only log at debug level on mismatch


class CouplingConfig:
    """
    Coupling configuration.
    """

    component_name: str  # Name of this component
    couple_to_icon_atmo: bool  # Whether to couple this component to ICON atmosphere
    couple_to_elmer_ice: bool  # Whether to couple this component to Elmer/Ice
    coupler_config: Optional[Path]  # Path to the coupler configuration file
    field_validation_level: FieldValidationLevel  # Level of validation for field exchange types

    def __init__(self, args: Namespace, component_name: str):
        """
        Initialize coupling configuration from command line arguments.

        @param[in] args command line arguments
        @param[in] component_name name of this component
        """

        self.component_name = component_name
        self.couple_to_icon_atmo = args.couple_to_icon_atmo
        self.couple_to_elmer_ice = args.couple_to_elmer_ice

        # Set field validation level from args (command-line argument with default 'FATAL')
        self.field_validation_level = FieldValidationLevel(args.field_validation_level)

        if args.coupler_config:
            assert Path(args.coupler_config).is_file(), f"Coupler configuration file {args.coupler_config} not found."
            self.coupler_config = args.coupler_config
        else:
            self.coupler_config = None
            logger.info(
                "No coupler configuration file provided. "
                "This is fine if configuration is provided by other components or through the API."
            )

    def defines_coupling(self) -> bool:
        """Check if any coupling is defined in this configuration.

        @returns True if coupling to any component is enabled, False otherwise
        """
        return self.couple_to_icon_atmo or self.couple_to_elmer_ice


class GridConfig:
    """
    Grid configuration.
    """

    grid_type: GridInputType  # Name of the grid used in coupling
    mesh_file: Path  # Path to the grid file
    dem_file: Optional[Path] = None  # Path to the DEM file (only relevant for CUSTOM grid type)
    is_partitioned: bool  # Whether the grid is partitioned
    is_unstructured: bool = False  # Whether the grid is unstructured
    partition_id: int  # Partition ID (only relevant if is_partitioned is True)
    elmer_mesh_crs_epsg: int  # EPSG code of Elmer mesh coordinates

    def __init__(self, args: Namespace):
        """
        Initialize grid configuration from command line arguments.

        @param[in] args command line arguments
        """
        if not (args.elmer_mesh or args.matlab_mesh):
            logger.error("Grid needed. Please provide either --elmer-mesh or --matlab-mesh.")
            raise Exception("Missing grid.")

        if args.elmer_mesh and args.matlab_mesh:
            logger.error("Please provide either --elmer-mesh or --matlab-mesh, not both.")
            raise Exception("Invalid grid configuration.")

        if args.is_partitioned_elmer_mesh and not args.elmer_mesh:
            logger.error("--is-partitioned-elmer-mesh requires --elmer-mesh.")
            raise Exception("Invalid grid configuration.")

        self.elmer_mesh_crs_epsg = args.elmer_mesh_crs_epsg

        self.is_partitioned = args.is_partitioned_elmer_mesh
        if self.is_partitioned:
            assert args.netcdf_mesh, (
                "--is-partitioned-elmer-mesh requires --netcdf-mesh. "
                "(Without --netcdf-mesh should also work but is untested.)"
            )
            logger.info("Using partitioned grid...")
            self.partition_id = args.use_part
            logger.info(f"{self.partition_id=}")
        else:
            logger.info("Using non-partitioned grid...")

        if args.matlab_mesh:
            self.grid_type = GridInputType.MATLAB
            self.mesh_file = args.matlab_mesh
        elif args.netcdf_mesh and args.elmer_mesh:
            self.grid_type = GridInputType.CUSTOM
            self.mesh_file = args.elmer_mesh
            self.dem_file = args.netcdf_mesh
        elif args.netcdf_mesh_unstructured and args.elmer_mesh:
            self.grid_type = GridInputType.ELMERXIOS
            self.mesh_file = args.elmer_mesh
            self.dem_file = args.netcdf_mesh_unstructured
            self.is_unstructured = True
        elif args.elmer_mesh:
            self.grid_type = GridInputType.ELMER
            self.mesh_file = args.elmer_mesh
        else:
            logger.error(
                f"Invalid grid configuration. EBFM supports the grid types {[t.name for t in GridInputType]}. "
                "Please refer to the documentation for correct configuration."
            )
            raise Exception("Invalid grid configuration.")


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

    def to_dict(self) -> dict:
        """Convert time configuration to a dictionary.

        @returns Dictionary representation of the time configuration
        """
        return {
            "ts": self.start_time,
            "te": self.end_time,
            "dt": self.time_step.total_seconds() / SECONDS_PER_DAY,  # Convert to days
            "tn": self.tn(),
            "dT_UTC": self.dT_UTC,
        }
