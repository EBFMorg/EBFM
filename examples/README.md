<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Example Files

| File                            | Description                                                         |
|-------------------------------- | ------------------------------------------------------------------- |
| `BedMachineGreenland-v5_lo.nc`  | Low-res version of `BedMachineGreenland-v5.nc`                      |
| `dem_and_mask.mat`              | MATLAB example grid                                                 |
| `greenland_mesh_v0/MESH`        | Elmer/Ice example mesh, Greenland (EPSG:3413)                       |
| `greenland_mesh_v0/xios_dem.nc` | Elevation and ice thickness on `greenland_mesh_v0` from Elmer/Ice   |
| `Greenland_CARRA2`              | 10-km Greenland grid and CARRA2 forcing for May 1990                |

See `Greenland_CARRA2/README.md` for how the files in `Greenland_CARRA2` were produced.

## How `BedMachineGreenland-v5_lo.nc` was produced

`BedMachineGreenland-v5_lo.nc` is a low-resolution subset of the
[`BedMachineGreenland-v5.nc`](https://nsidc.org/data/idbmg4/versions/5) dataset (Morlighem, 2022) provided as an
alternative if you cannot or do not want to download the rather large dataset from the original source.

The low-resolution file only contains the `surface` field (ice surface elevation, EPSG:3413) and subsamples the
original 150 m grid by a factor of 10, giving a 1022 x 1835 point grid at ~1500 m effective resolution. To reproduce
from the original file, please download [`BedMachineGreenland-v5.nc`](https://nsidc.org/data/idbmg4/versions/5) and
then run:

```bash
python3 tools/nc_reduce_size.py \
    BedMachineGreenland-v5.nc \
    --keep-only surface \
    --stride 10 \
    -o examples/BedMachineGreenland-v5_lo.nc
```

See `tools/README.md` for full documentation of `nc_reduce_size.py`.

## How `greenland_mesh_v0/xios_dem.nc` was produced

`greenland_mesh_v0/xios_dem.nc` provides the surface elevation and ice thickness for the mesh `greenland_mesh_v0/MESH`
as Elmer/Ice writes them through XIOS. It is the input for `--netcdf-dem-mesh-unstructured`:

```bash
ebfm --elmer-mesh examples/greenland_mesh_v0/MESH \
    --netcdf-dem-mesh-unstructured examples/greenland_mesh_v0/xios_dem.nc \
    --elmer-mesh-crs-epsg 3413
```

EBFM reads two variables from the file:

* `zs`: surface elevation (m)
* `h`: ice thickness (m)

Both have the dimensions `(time, nmesh2D_node)` with a single time step and hold one value per mesh node, in the order
of `mesh.nodes`. EBFM averages the three node values of each element to obtain the elevation and ice thickness of the
corresponding column and treats columns with more than 1 m of ice as glacier. The file also contains the node
coordinates `x` and `y` (EPSG:3413), which agree with `mesh.nodes` and allow to check the node order.

The original file was written by Elmer/Ice v9.0 (Rev: c4a78ac59) with the `XIOSOutputSolver`. The relevant part of
the XIOS context is:

```xml
<field_group id="elmer_topo" >
    <field id="h"  standard_name="land_ice_thickness" unit="m" grid_ref="GridNodes" />
    <field id="zs" standard_name="surface_altitude"   unit="m" grid_ref="GridNodes" />
</field_group>
<field_group id="mesh_info2" operation="instant" >
    <field id="node_x" name="x" standard_name="projection_x_coordinate" unit="m" grid_ref="GridNodes" />
    <field id="node_y" name="y" standard_name="projection_y_coordinate" unit="m" grid_ref="GridNodes" />
</field_group>
...
<file_definition type="one_file" sync_freq="24h">
    <file id="elmer_restart" name="restart" convention="UGRID" output_freq="24h" time_counter_name="time"
          time_counter="instant" time_units="days" >
        <field_group group_ref="mesh_info2" />
        <field_group group_ref="elmer_topo" />
    </file>
</file_definition>
```

Besides these fields, XIOS writes the mesh topology of the UGRID convention, which EBFM takes from the Elmer mesh
instead. `xios_dem.nc` only keeps the node fields and compresses them (2.5 MB instead of 39 MB):

```bash
nccopy -d 4 -s -V zs,h,x,y restart.nc examples/greenland_mesh_v0/xios_dem.nc
```
