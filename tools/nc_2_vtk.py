# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert 2-D fields from a NetCDF file to a VTK legacy binary rectilinear grid.

Each selected field is written as a named point-data array on a flat (z=0) grid
using the native projected x/y coordinates from the file.  The output can be
opened directly in ParaView or any other VTK-capable visualisation tool.

Grid infrastructure variables (x, y) are always read and used as grid axes.
"""

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# Variables treated as grid axes, not data fields
GRID_VARS = {"x", "y", "mapping"}


def netcdf_to_vtk(
    nc_path: Path,
    out_path: Path,
    fields: list[str] | None = None,
    stride: int = 1,
) -> None:
    """Read 2-D fields from a NetCDF file and write a VTK legacy binary rectilinear grid.

    Args:
        nc_path:  Path to the input NetCDF file.
        out_path: Path for the output .vtk file.
        fields:   List of 2-D variable names to embed as point-data arrays.
                  None means include all 2-D fields.
        stride:   Subsampling stride in both x and y (e.g. 10 -> every 10th point).
    """
    with netCDF4.Dataset(nc_path) as nc:
        all_fields = sorted(
            name for name, v in nc.variables.items() if name not in GRID_VARS and set(v.dimensions) >= {"x", "y"}
        )

        if fields is not None:
            unknown = [f for f in fields if f not in all_fields]
            if unknown:
                raise ValueError(
                    f"field(s) not found in {nc_path.name}: {unknown}\n" f"Available 2-D fields: {all_fields}"
                )
            selected_fields = fields
        else:
            selected_fields = all_fields

        x = np.asarray(nc["x"][::stride], dtype=np.float64)
        y = np.asarray(nc["y"][::stride], dtype=np.float64)
        data = {name: np.asarray(nc[name][::stride, ::stride], dtype=np.float32) for name in selected_fields}

    # z-axis is a single plane at z=0; field values are stored as point data.
    z = np.array([0.0], dtype=np.float64)

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(len(x), len(y), 1)
    grid.SetXCoordinates(numpy_to_vtk(x, deep=True, array_type=vtk.VTK_DOUBLE))
    grid.SetYCoordinates(numpy_to_vtk(y, deep=True, array_type=vtk.VTK_DOUBLE))
    grid.SetZCoordinates(numpy_to_vtk(z, deep=True, array_type=vtk.VTK_DOUBLE))

    for i, (name, arr) in enumerate(data.items()):
        # VTK expects points ordered x-fastest, y-next (row-major / C order).
        vtk_array = numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(name)
        if i == 0:
            grid.GetPointData().SetScalars(vtk_array)
        else:
            grid.GetPointData().AddArray(vtk_array)

    # Use legacy binary format - robust across VTK/ParaView version differences.
    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    print(f"Written : {out_path}")
    print(f"  Fields  : {selected_fields}")
    print(f"  Grid    : {len(x)} x {len(y)} points  (stride {stride})")
    print(f"  x range : {x.min():.0f} - {x.max():.0f} m")
    print(f"  y range : {y.min():.0f} - {y.max():.0f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 2-D NetCDF fields to a VTK rectilinear grid for visualisation in ParaView.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Export the surface field at full resolution\n"
            "  %(prog)s input.nc --keep-only surface\n\n"
            "  # Export surface and bed at 10x reduced resolution (fast preview)\n"
            "  %(prog)s input.nc --keep-only surface bed --stride 10\n\n"
            "  # Export all 2-D fields\n"
            "  %(prog)s input.nc\n"
        ),
    )
    parser.add_argument("nc_file", type=Path, help="Input NetCDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .vtk path (default: input stem + .vtk in the same directory).",
    )
    parser.add_argument(
        "--keep-only",
        nargs="+",
        metavar="FIELD",
        default=None,
        help=(
            "2-D field(s) to include in the VTK file (e.g. --keep-only surface bed). "
            "Omit to include all 2-D fields found in the file."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Subsample every N-th point in x and y, reducing point count by N². "
            "Useful for fast previews (e.g. --stride 10 gives ~100x fewer points). "
            "Default: 1 (full resolution)."
        ),
    )
    args = parser.parse_args()

    out = args.output if args.output is not None else args.nc_file.with_suffix(".vtk")

    try:
        netcdf_to_vtk(args.nc_file, out, fields=args.keep_only, stride=args.stride)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
