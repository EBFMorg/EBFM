# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for EBFM.

This file exposes a some configuration dataclasses for EBFM components.
"""

from argparse import Namespace
from pathlib import Path

from ebfm.grid import GridInputType

import logging

logger = logging.getLogger(__name__)


class CouplingConfig:
    """
    Coupling configuration.
    """

    component_name: str  # Name of this component
    couple_to_icon_atmo: bool  # Whether to couple this component to ICON atmosphere
    couple_to_elmer_ice: bool  # Whether to couple this component to Elmer/Ice
    coupler_config: Path  # Path to the coupler configuration file

    def __init__(self, args: Namespace, component_name: str):
        """
        Initialize coupling configuration from command line arguments.

        @param[in] args command line arguments
        @param[in] component_name name of this component
        """

        self.component_name = component_name
        self.couple_to_icon_atmo = args.couple_to_icon_atmo
        self.couple_to_elmer_ice = args.couple_to_elmer_ice

        if self.defines_coupling() and not args.coupler_config:
            logger.error("Coupling enabled but no coupler configuration file provided (--coupler-config).")
            raise Exception("Missing coupler configuration file.")

        self.coupler_config = args.coupler_config

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
    dem_file: Path = None  # Path to the DEM file (only relevant for CUSTOM grid type)
    is_partitioned: bool  # Whether the grid is partitioned
    partition_id: int  # Partition ID (only relevant if is_partitioned is True)

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
            self.grid_type = GridInputType.XIOS_CUSTOM
            self.mesh_file = args.elmer_mesh
            self.dem_file = args.netcdf_mesh_unstructured
        elif args.elmer_mesh:
            self.grid_type = GridInputType.ELMER
            self.mesh_file = args.elmer_mesh
        else:
            logger.error(
                f"Invalid grid configuration. EBFM supports the grid types {[t.name for t in GridInputType]}. "
                "Please refer to the documentation for correct configuration."
            )
            raise Exception("Invalid grid configuration.")
