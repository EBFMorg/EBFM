# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum


class GridInputType(Enum):
    # .mat grid file with elevation
    MATLAB = "matlab"

    # Elmer/Ice mesh file for xy-coordinates and separate NetCDF elevation file
    CUSTOM = "custom"

    # Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS
    XIOS_CUSTOM = "xios_custom"

    # Elmer/Ice mesh file with elevation in z-coordinate
    ELMER = "elmer"
