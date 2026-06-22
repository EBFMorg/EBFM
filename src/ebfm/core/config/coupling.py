# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Coupling Configuration for EBFM.

This file exposes the coupling configuration dataclass for EBFM components.
"""

from argparse import Namespace
from pathlib import Path
from enum import Enum
from mpi4py import MPI

from datetime import datetime

from .time import TimeConfig

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
        time_config: TimeConfig,
    ):
        """
        Initialize coupling configuration from command line arguments.

        @param[in] args command line arguments
        @param[in] time_config time configuration
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

        # Set start and end time used by coupled (must be consistent with the one used by this component)
        self.time_config = time_config

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

    @property
    def start_time(self) -> datetime:
        """Get the start time of the simulation."""
        return self.time_config.start_time

    @property
    def end_time(self) -> datetime:
        """Get the end time of the simulation."""
        return self.time_config.end_time

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
