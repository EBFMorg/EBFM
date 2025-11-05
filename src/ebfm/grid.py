from enum import Enum


class GridInputType(Enum):
    MATLAB = "matlab"  # .mat grid file with elevation
    CUSTOM = "custom"  # Elmer/Ice mesh file for xy-coordinates and separate NetCDF elevation file
    ELMER = "elmer"  # Elmer/Ice mesh file with elevation in z-coordinate
