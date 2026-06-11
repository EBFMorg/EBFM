# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reduce the size of a BedMachine-style NetCDF file.

Produces a smaller copy by optionally dropping unwanted data fields and/or
subsampling the x/y grid at a given stride.  Global attributes, the CRS
variable (mapping), and the grid axes (x, y) are always preserved so the
output remains a fully self-describing NetCDF file.
"""

import argparse
import sys
from pathlib import Path

import netCDF4

# Always-kept infrastructure variables (CRS + grid axes)
GRID_VARS = {"mapping", "x", "y"}


def strip_netcdf(
    in_path: Path,
    out_path: Path,
    keep_only: list[str] | None = None,
    stride: int = 1,
) -> None:
    """Copy selected fields from a NetCDF file to a new, smaller file.

    Args:
        in_path:   Path to the source NetCDF file.
        out_path:  Path for the output NetCDF file.
        keep_only: Data field names to keep. None means keep all data fields.
        stride:    Subsampling stride applied to both x and y axes.
    """
    with netCDF4.Dataset(in_path, "r") as src:
        # Determine which variables are "data fields" (2-D, on the x/y grid)
        all_fields = sorted(
            name for name, v in src.variables.items() if name not in GRID_VARS and set(v.dimensions) >= {"x", "y"}
        )

        if keep_only is not None:
            unknown = [f for f in keep_only if f not in all_fields]
            if unknown:
                raise ValueError(
                    f"field(s) not found in {in_path.name}: {unknown}\n" f"Available 2-D fields: {all_fields}"
                )
            fields_to_write = keep_only
        else:
            fields_to_write = all_fields

        keep_vars = GRID_VARS | set(fields_to_write)

        with netCDF4.Dataset(out_path, "w", format=src.file_format) as dst:
            # Global attributes — copy originals, then append a provenance note
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
            original_name = in_path.name
            kept = ", ".join(sorted(fields_to_write))
            dst.history = (
                f"Reduced from {original_name} using tools/nc_reduce_size.py from https://github.com/EBFMorg/EBFM."
                f" (stride={stride}, kept fields: {kept})" + (getattr(dst, "history", "") or "")
            ).strip()

            # Dimensions - resized when stride > 1
            for name, dim in src.dimensions.items():
                if name in ("x", "y"):
                    new_len = len(src.variables[name][::stride])
                    dst.createDimension(name, new_len)
                else:
                    dst.createDimension(name, None if dim.isunlimited() else len(dim))

            # Write selected variables
            for name in keep_vars:
                if name not in src.variables:
                    continue
                sv = src.variables[name]
                attrs = {a: sv.getncattr(a) for a in sv.ncattrs() if a != "_FillValue"}
                fill = sv.getncattr("_FillValue") if "_FillValue" in sv.ncattrs() else None

                dv = dst.createVariable(name, sv.dtype, sv.dimensions, fill_value=fill)
                dv.setncatts(attrs)

                # Apply stride on x/y dimensions
                if sv.dimensions == ("y", "x"):
                    dv[:] = sv[::stride, ::stride]
                elif sv.dimensions == ("y",):
                    dv[:] = sv[::stride]
                elif sv.dimensions == ("x",):
                    dv[:] = sv[::stride]
                else:
                    dv[:] = sv[:]

        # Collect reporting info before src closes
        src_x_len = len(src.variables["x"][::stride])
        src_y_len = len(src.variables["y"][::stride])
        total_size = sum(v[:].nbytes for v in src.variables.values())
        dropped = sorted(set(src.variables) - keep_vars)

    with netCDF4.Dataset(out_path, "r") as dst:
        out_size = sum(v[:].nbytes for v in dst.variables.values())

    print(f"Written: {out_path}")
    print(f"  Kept fields    : {sorted(fields_to_write)}")
    if dropped:
        print(f"  Dropped fields : {dropped}")
    print(f"  Stride         : {stride}  ->  grid {src_x_len} x {src_y_len}")
    print(f"  Data size      : {out_size / 1e6:.1f} MB  (was {total_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce a smaller copy of a BedMachine-style NetCDF file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Keep only the surface field at full resolution\n"
            "  %(prog)s input.nc --keep-only surface\n\n"
            "  # Keep surface and bed at 10x reduced resolution\n"
            "  %(prog)s input.nc --keep-only surface bed --stride 10\n\n"
            "  # Keep all fields but reduce grid resolution by 5x\n"
            "  %(prog)s input.nc --stride 5\n"
        ),
    )
    parser.add_argument("input", type=Path, help="Source NetCDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output NetCDF path (default: <input stem>_lofi.nc in the same directory).",
    )
    parser.add_argument(
        "--keep-only",
        nargs="+",
        metavar="FIELD",
        default=None,
        help=(
            "Data field(s) to retain (e.g. --keep-only surface bed). "
            "Grid variables (x, y, mapping) are always kept. "
            "Omit to retain all data fields."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Subsample every N-th point in x and y, reducing point count by N*N. "
            "E.g. --stride 10 gives ~100X fewer grid points. "
            "Default: 1 (full resolution)."
        ),
    )
    args = parser.parse_args()

    out = args.output if args.output is not None else args.input.with_stem(args.input.stem + "_reduced")
    if out.exists():
        print(f"Error: output file already exists: {out}", file=sys.stderr)
        sys.exit(1)

    try:
        strip_netcdf(args.input, out, keep_only=args.keep_only, stride=args.stride)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
