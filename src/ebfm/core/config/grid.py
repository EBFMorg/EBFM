# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Grid configuration for EBFM.

This file provides the grid configuration dataclass for EBFM.
"""

from argparse import Namespace

from ebfm.core.grid import GridInputType
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


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

        # Shading is only supported for MATLAB meshes; see https://github.com/EBFMorg/EBFM/issues/11
        grid_type_supports_shading_supported = self.grid_type is GridInputType.MATLAB

        # Partitioned grids don't support shading
        grid_partitioning_supports_shading = not self.is_partitioned

        # shading is supported if both the grid type and the partitioning support it
        _shading_supported = grid_type_supports_shading_supported and grid_partitioning_supports_shading

        if args.shading is None:
            self.use_shading = _shading_supported  # default: on for MATLAB, off for all others
        else:
            if args.shading and not _shading_supported:
                if not grid_type_supports_shading_supported:
                    raise ValueError(
                        f"Shading is not supported for grid type {self.grid_type}. "
                        "See https://github.com/EBFMorg/EBFM/issues/11"
                    )
                if not grid_partitioning_supports_shading:
                    raise ValueError("Shading is not supported for partitioned grids.")

            self.use_shading = args.shading
