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

    # maps GridInputType to corresponding argparse destination
    mesh_arg_dests = {
        GridInputType.ELMER: "elmer_mesh",
        GridInputType.MATLAB: "matlab_mesh",
        GridInputType.GREENLAND: "greenland_mesh",
    }

    # Shading is only supported for some grid types
    grid_types_supporting_shading = {
        GridInputType.MATLAB,
    }

    def __init__(self, args: Namespace):
        """
        Initialize grid configuration from command line arguments.

        @param[in] args command line arguments
        """
        selected_primary_grids = [
            grid_type
            for grid_type, arg_dest in self.mesh_arg_dests.items()
            if getattr(args, arg_dest, None) is not None
        ]
        assert len(selected_primary_grids) == 1, "Internal error: expected exactly one primary grid option to be set."

        matlab_mesh = getattr(args, self.mesh_arg_dests[GridInputType.MATLAB], None)
        greenland_mesh = getattr(args, self.mesh_arg_dests[GridInputType.GREENLAND], None)
        elmer_mesh = getattr(args, self.mesh_arg_dests[GridInputType.ELMER], None)

        self.elmer_mesh_crs_epsg = args.elmer_mesh_crs_epsg

        self.is_partitioned = args.is_partitioned_elmer_mesh
        self.dem_file = None
        self.partition_id = None

        assert (
            not self.is_partitioned or elmer_mesh is not None
        ), "Internal error: partitioned grid configuration requires an Elmer mesh input."

        if self.is_partitioned:
            assert args.netcdf_mesh, "Internal error: partitioned grid configuration requires NetCDF mesh input."
            logger.info("Using partitioned grid...")
            self.partition_id = args.use_part
            logger.info(f"{self.partition_id=}")
        else:
            logger.info("Using non-partitioned grid...")

        if matlab_mesh:
            self.grid_type = GridInputType.MATLAB
            self.mesh_file = matlab_mesh
            self.is_unstructured = False
        elif greenland_mesh:
            self.grid_type = GridInputType.GREENLAND
            self.mesh_file = greenland_mesh
            self.is_unstructured = False
        elif args.netcdf_mesh and elmer_mesh:
            self.grid_type = GridInputType.CUSTOM
            self.mesh_file = elmer_mesh
            self.dem_file = args.netcdf_mesh
            self.is_unstructured = False
        elif args.netcdf_mesh_unstructured and elmer_mesh:
            self.grid_type = GridInputType.ELMERXIOS
            self.mesh_file = elmer_mesh
            self.dem_file = args.netcdf_mesh_unstructured
            self.is_unstructured = True
        elif elmer_mesh:
            self.grid_type = GridInputType.ELMER
            self.mesh_file = elmer_mesh
            self.is_unstructured = False
        else:
            logger.error(
                f"Invalid grid configuration. EBFM supports the grid types {[t.name for t in GridInputType]}. "
                "Please refer to the documentation for correct configuration."
            )
            raise Exception("Invalid grid configuration.")

        self.use_shading = self._check_shading(args)

    def _check_shading(self, args: Namespace) -> bool:
        """Configure shading based on the user input, grid type, and partitioning."""
        grid_type_supports_shading = self.grid_type in self.grid_types_supporting_shading

        # Partitioned grids don't support shading
        grid_partitioning_supports_shading = not self.is_partitioned

        # shading is supported if both the grid type and the partitioning support it
        shading_supported = grid_type_supports_shading and grid_partitioning_supports_shading

        # if user did not explicitly set the shading option use the default based on the grid configuration
        if not hasattr(args, "shading"):
            return shading_supported

        if not shading_supported and args.shading:
            if not grid_type_supports_shading:
                raise ValueError(
                    f"Shading is not supported for grid type {self.grid_type}. "
                    "See https://github.com/EBFMorg/EBFM/issues/129"
                )
            if not grid_partitioning_supports_shading:
                raise ValueError("Shading is not supported for partitioned grids.")

        return args.shading
