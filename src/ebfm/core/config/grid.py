# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""Grid configuration for EBFM.

This file provides the grid configuration dataclass for EBFM.
"""

from argparse import Namespace

from ebfm.core.grid import GridInputType
from ebfm.grid.structured.partitioning import partition_file
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
    partition_id: int | None  # Partition ID (only relevant if is_partitioned is True or num_partitions > 1)
    # Number of partitions of the grid, one per MPI rank; 1 for a grid that is not partitioned
    num_partitions: int
    elmer_mesh_crs_epsg: int  # EPSG code of Elmer mesh coordinates
    use_shading: bool  # Whether to use shading for the grid
    fill_missing_elevation: bool  # Whether to set missing elevations to 0 m (only relevant for NETCDF grid type)

    # UTM zone of grid coordinates, used to derive lat lon fields
    utmzone: int = 33

    # maps GridInputType to corresponding argparse destination
    mesh_arg_dests = {
        GridInputType.ELMER: "elmer_mesh",
        GridInputType.MATLAB: "matlab_mesh",
        GridInputType.NETCDF: "netcdf_mesh",
    }

    # Shading is only supported for some grid types
    grid_types_supporting_shading = {
        GridInputType.MATLAB,
    }

    def __init__(self, args: Namespace):
        """
        Initialize grid configuration from command line arguments.

        @param[in] args command line arguments, with args.partitioning set by cli.resolve_partitioning in a run on
                        several MPI ranks
        """
        selected_primary_grids = [
            grid_type
            for grid_type, arg_dest in self.mesh_arg_dests.items()
            if getattr(args, arg_dest, None) is not None
        ]
        assert len(selected_primary_grids) == 1, "Internal error: expected exactly one primary grid option to be set."

        matlab_mesh = getattr(args, self.mesh_arg_dests[GridInputType.MATLAB], None)
        netcdf_mesh = getattr(args, self.mesh_arg_dests[GridInputType.NETCDF], None)
        elmer_mesh = getattr(args, self.mesh_arg_dests[GridInputType.ELMER], None)

        self.elmer_mesh_crs_epsg = args.elmer_mesh_crs_epsg

        partitioning = getattr(args, "partitioning", None)
        self.is_partitioned = args.is_partitioned_elmer_mesh or partitioning is not None
        self.dem_file = None
        self.partition_id = None
        self.num_partitions = 1

        if matlab_mesh:
            self.grid_type = GridInputType.MATLAB
            self.mesh_file = matlab_mesh
            self.is_unstructured = False
        elif netcdf_mesh:
            self.grid_type = GridInputType.NETCDF
            self.mesh_file = netcdf_mesh
            self.dem_file = netcdf_mesh  # netcdf_mesh includes DEM data
            self.is_unstructured = False
        elif args.netcdf_dem_mesh and elmer_mesh:
            self.grid_type = GridInputType.CUSTOM
            self.mesh_file = elmer_mesh
            self.dem_file = args.netcdf_dem_mesh
            self.is_unstructured = True
        elif args.netcdf_dem_mesh_unstructured and elmer_mesh:
            self.grid_type = GridInputType.ELMERXIOS
            self.mesh_file = elmer_mesh
            self.dem_file = args.netcdf_dem_mesh_unstructured
            self.is_unstructured = True
        elif elmer_mesh:
            self.grid_type = GridInputType.ELMER
            self.mesh_file = elmer_mesh
            self.is_unstructured = True
        else:
            logger.error(
                f"Invalid grid configuration. EBFM supports the grid types {[t.name for t in GridInputType]}. "
                "Please refer to the documentation for correct configuration."
            )
            raise Exception("Invalid grid configuration.")

        if self.is_partitioned:
            self.partition_id = args.use_part
            if self.is_unstructured:
                self.num_partitions = len(list(Path(elmer_mesh).glob("part.*.header")))
            else:
                # A partition of a structured grid is a grid file of its own, read instead of the whole grid
                self.num_partitions = len(list(Path(partitioning).glob(f"part.*{Path(self.mesh_file).suffix}")))
                self.mesh_file = partition_file(partitioning, self.partition_id, self.mesh_file)
                if self.grid_type is GridInputType.NETCDF:
                    self.dem_file = self.mesh_file
            logger.info(f"Using partition {self.partition_id} of {self.num_partitions}...")

        self.use_shading = self._check_shading(args)
        self.fill_missing_elevation = args.fill_missing_elevation

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
