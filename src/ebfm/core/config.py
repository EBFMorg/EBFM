# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for EBFM.

This file exposes a some configuration dataclasses for EBFM components.
"""

from argparse import Namespace
from pathlib import Path
from enum import Enum
from mpi4py import MPI

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


class ComponentId(Enum):
    """
    Enumeration of component identifiers for EBFM coupling.

    Lists identifiers for available components without having to import the component classes directly which would lead
    to a circular import.
    """

    ICON_ATMO = "icon_atmo"
    ELMER_ICE = "elmer_ice"


class CouplingConfig:
    """
    Coupling configuration.
    """

    component_name: str  # Name of this component
    coupler_config: Path | None  # Path to the coupler configuration file
    field_validation_level: FieldValidationLevel  # Level of validation for field exchange types
    use_fake_coupling: bool  # Whether to use FakeCoupler instead of production backend
    comms: dict[str, MPI.Comm] | None  # Dict with MPI communicators used by this model

    def __init__(
        self,
        args: Namespace,
    ):
        """
        Initialize coupling configuration from command line arguments.

        @param[in] args command line arguments
        """

        self.component_name = args.component_name

        self._coupled_components = {
            ComponentId.ICON_ATMO: args.couple_to_icon_atmo,
            ComponentId.ELMER_ICE: args.couple_to_elmer_ice,
        }

        self.use_fake_coupling = args.fake_coupling

        # will be defined later because we already need some members of CouplingConfig to create the communicators
        self.comms = None

        # Set field validation level from args (command-line argument with default 'FATAL')
        self.field_validation_level = FieldValidationLevel(args.field_validation_level)

        if args.coupler_config:
            if not Path(args.coupler_config).is_file():
                raise FileNotFoundError(f"Coupler configuration file {args.coupler_config} not found.")
            self.coupler_config = args.coupler_config
        else:
            self.coupler_config = None
            logger.info(
                "No coupler configuration file provided. "
                "This is fine if configuration is provided by other components or through the API."
            )

    def set_group_communicators(self, group_comms: dict[str, MPI.Comm]):
        """Store group communicators in the coupling configuration.

        @param[in] group_comms dict with MPI communicators for different groups
        """
        if self.has_group_communicators():
            raise RuntimeError(
                "Group communicators have already been set. Overwriting them is not allowed since "
                "mpi-handshake should only be performed once and this hints at a programming error in your code."
            )
        self.comms = group_comms

    def has_group_communicators(self) -> bool:
        """Check if group communicators have been set.

        @returns True if group communicators are set, False otherwise
        """
        return self.comms is not None

    def has_group_communicator(self, group_label: str) -> bool:
        """Check if a communicator for the given group label is available.

        @param[in] group_label label of the group
        @returns True if a communicator for the group label is available, False otherwise
        """
        return self.has_group_communicators() and group_label in self.comms  # type: ignore[operator]

    def get_group_communicator(self, group_label: str) -> MPI.Comm:
        """Get the communicator for the given group label.

        @param[in] group_label label of the group
        @returns MPI communicator for the group label
        """
        if not self.has_group_communicator(group_label):
            raise KeyError(f"Communicator for group label '{group_label}' not found.")
        return self.comms[group_label]  # type: ignore[index]

    def _active_coupling_to(self, component_id: ComponentId) -> bool:
        """Check if coupling to a specific component is enabled.

        @note Asserts that the component name exists in self._coupled_components. This prevents forgetting to register
              a component during initialization.

        @param[in] component_name name of the component to check coupling for

        @returns True if coupling to the specified component is enabled, False if disabled
        """
        assert (
            component_id in self._coupled_components
        ), f"Coupling configuration is missing {component_id} entry. Make sure to explicitly set True/False for the "
        "given key in the __init__ method of CouplingConfig."
        return self._coupled_components[component_id]

    @property
    def couple_to_icon_atmo(self):
        """Whether to couple this component to ICON atmosphere."""
        return self._active_coupling_to(ComponentId.ICON_ATMO)

    @property
    def couple_to_elmer_ice(self):
        """Whether to couple this component to Elmer/Ice."""
        return self._active_coupling_to(ComponentId.ELMER_ICE)

    def active_coupled_components(self) -> list[ComponentId]:
        """Get a list of actively coupled components based on the configuration.

        @returns List of ComponentId for which coupling is enabled
        """
        return [c for c in ComponentId if self._active_coupling_to(c)]

    def defines_coupling(self) -> bool:
        """Check if any coupling is defined in this configuration.

        @returns True if coupling to any component is enabled, False otherwise
        """
        return len(self.active_coupled_components()) > 0


class GridConfig:
    """
    Grid configuration.
    """

    grid_type: GridInputType  # Name of the grid used in coupling
    mesh_file: Path  # Path to the grid file
    dem_file: Path | None  # Path to the DEM file (only relevant for CUSTOM grid type)
    is_partitioned: bool  # Whether the grid is partitioned
    is_unstructured: bool  # Whether the grid is unstructured
    partition_id: int | None  # Partition ID (only relevant if is_partitioned is True)
    elmer_mesh_crs_epsg: int  # EPSG code of Elmer mesh coordinates
    use_shading: bool  # Whether to use shading for the grid

    # maps GridInputType to corresponding command line argument
    mesh_opts = {
        GridInputType.ELMER: "--elmer-mesh",
        GridInputType.MATLAB: "--matlab-mesh",
        GridInputType.GREENLAND: "--greenland-mesh",
    }

    # Shading is only supported for some grid types
    grid_types_supporting_shading = {
        GridInputType.MATLAB,
        GridInputType.GREENLAND,
    }

    def __init__(self, args: Namespace):
        """
        Initialize grid configuration from command line arguments.

        @param[in] args command line arguments
        """

        provided_grid = {
            GridInputType.ELMER: args.elmer_mesh is not None,
            GridInputType.MATLAB: args.matlab_mesh is not None,
            GridInputType.GREENLAND: args.greenland_mesh is not None,
        }

        if sum(provided_grid.values()) > 1:
            logger.error(
                "Providing more than one grid is forbidden. "
                "You are currently providing the following incompatible options: "
                f"{', '.join([GridConfig.mesh_opts[g] for g in provided_grid if provided_grid[g]])}."
            )
            raise Exception("Invalid grid configuration.")

        if sum(provided_grid.values()) == 0:
            logger.error(
                "No grid provided. "
                "Please provide exactly one grid using one of the following options: "
                f"{', '.join(self.mesh_opts.values())}."
            )
            raise Exception("Invalid grid configuration.")

        if args.is_partitioned_elmer_mesh and not args.elmer_mesh:
            logger.error(f"--is-partitioned-elmer-mesh requires {self.mesh_opts[GridInputType.ELMER]}.")
            raise Exception("Invalid grid configuration.")

        self.elmer_mesh_crs_epsg = args.elmer_mesh_crs_epsg

        self.is_partitioned = args.is_partitioned_elmer_mesh
        self.dem_file = None
        self.partition_id = None

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
            self.is_unstructured = False
        elif args.greenland_mesh:
            self.grid_type = GridInputType.GREENLAND
            self.mesh_file = args.greenland_mesh
            self.is_unstructured = False
        elif args.netcdf_mesh and args.elmer_mesh:
            self.grid_type = GridInputType.CUSTOM
            self.mesh_file = args.elmer_mesh
            self.dem_file = args.netcdf_mesh
            self.is_unstructured = False
        elif args.netcdf_mesh_unstructured and args.elmer_mesh:
            self.grid_type = GridInputType.ELMERXIOS
            self.mesh_file = args.elmer_mesh
            self.dem_file = args.netcdf_mesh_unstructured
            self.is_unstructured = True
        elif args.elmer_mesh:
            self.grid_type = GridInputType.ELMER
            self.mesh_file = args.elmer_mesh
            self.is_unstructured = False
        else:
            logger.error(
                f"Invalid grid configuration. EBFM supports the grid types {[t.name for t in GridInputType]}. "
                "Please refer to the documentation for correct configuration."
            )
            raise Exception("Invalid grid configuration.")

        self.use_shading = self._check_shading(args)

    def _check_shading(self, args: Namespace) -> bool:
        """Configure shading based on the user input, grid type, and partitioning.

        Shading is only supported for certain configurations, see https://github.com/EBFMorg/EBFM/issues/129.
        """
        grid_type_supports_shading = self.grid_type in self.grid_types_supporting_shading

        # Partitioned grids don't support shading
        grid_partitioning_supports_shading = not self.is_partitioned

        # shading is supported if both the grid type and the partitioning support it
        shading_supported = grid_type_supports_shading and grid_partitioning_supports_shading

        # if user did not explicitly set the shading option use the default based on the grid configuration
        if not hasattr(args, "shading"):
            return shading_supported  # default: use shading if supported, otherwise disable it

        assert hasattr(args, "shading"), "args should have 'shading' attribute at this point."

        if not shading_supported and args.shading:
            # user asks for shading but grid configuration does not support it
            if not grid_type_supports_shading:
                raise ValueError(
                    f"Shading is not supported for grid type {self.grid_type}. "
                    "See https://github.com/EBFMorg/EBFM/issues/129"
                )
            if not grid_partitioning_supports_shading:
                raise ValueError("Shading is not supported for partitioned grids.")

        # provided shading configuration is safe
        return args.shading


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

        self.dT_UTC = -3  # Time difference relative to UTC in hours (hard-coded for now)

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
