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
    ELMERXIOS = "elmerxios"

    # Elmer/Ice mesh file with elevation in z-coordinate
    ELMER = "elmer"


class ShadingMethod(Enum):
    """Available shading algorithms for topographic shading.

    - `CLASSICAL`: computes shading online each time step by ray-marching from
        each glacier cell in the current solar azimuth direction.
    - `LUT`: uses a precomputed look-up table of maximum horizon angles per cell
        and azimuth sector, which is faster at runtime.
    """

    CLASSICAL = "classical_shading"
    LUT = "lut_shading"
