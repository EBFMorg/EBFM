<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Tools

Utility scripts for working with BedMachine-style NetCDF files.

---

## `nc_reduce_size.py` - shrink a NetCDF file

Produces a smaller copy of a NetCDF file by dropping unwanted data fields
and/or subsampling the grid.  Global attributes, the CRS (`mapping`), and the
grid axes (`x`, `y`) are always preserved so the output remains fully
self-describing.

```bash
# Keep only the surface field (3.6 GB -> 750 MB)
python tools/nc_reduce_size.py input.nc --keep-only surface

# Keep surface and bed at 10x reduced resolution (3.6 GB -> ~15 MB)
python tools/nc_reduce_size.py input.nc --keep-only surface bed --stride 10

# Reduce grid resolution for all fields without dropping any
python tools/nc_reduce_size.py input.nc --stride 5

# Custom output path
python tools/nc_reduce_size.py input.nc --keep-only surface --stride 10 -o lofi.nc
```

---

## `nc_2_vtk.py` - convert NetCDF fields to VTK for ParaView

Reads one or more 2-D fields from a NetCDF file and writes a VTK legacy binary
rectilinear grid (`.vtk`) using the native projected x/y coordinates.  Each
field becomes a named point-data array, selectable in ParaView.

```bash
# Export the surface field at full resolution
python tools/nc_2_vtk.py input.nc --keep-only surface

# Fast preview: surface + bed at 10x reduced resolution (~2 M points)
python tools/nc_2_vtk.py input.nc --keep-only surface bed --stride 10

# Export all 2-D fields at full resolution
python tools/nc_2_vtk.py input.nc

# Custom output path
python tools/nc_2_vtk.py input.nc --keep-only surface --stride 10 -o surface.vtk
```

Open the resulting `.vtk` file in ParaView:

```bash
paraview surface.vtk
```
