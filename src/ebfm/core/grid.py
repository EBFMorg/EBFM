# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from typing import Any


class GridInputType(Enum):
    # .mat grid file with elevation
    MATLAB = "matlab"

    # Elmer/Ice mesh file for xy-coordinates and separate NetCDF elevation file
    CUSTOM = "custom"

    # Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS
    ELMERXIOS = "elmerxios"

    # Elmer/Ice mesh file with elevation in z-coordinate
    ELMER = "elmer"


GridDict = dict[str, Any]  # Alias for grid dictionary type, can be replaced with a more specific type in the future
