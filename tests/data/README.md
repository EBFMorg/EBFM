<!--
SPDX-FileCopyrightText: 2026 The EBFM Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Test data

`elmer_mesh` is a small synthetic Elmer mesh for tests that run EBFM end to end:

* `MESH`: a rectangle of 12 x 8 squares of 5 km, two triangles each, in EPSG:3413, with its lower left corner at
  x = -200000 m, y = -2000000 m. Node ids (117) and element ids (192) run from 1 in file order. There is no
  `mesh.boundary`, which EBFM does not read.
* `xios_dem.nc`: surface elevation `zs = 500 + 7 * id` and ice thickness `h` per node, stored by node id like Elmer/Ice
  XIOS output (see `examples/README.md`). `h` is 0.5 m for the first third of the node ids and `100 + id` otherwise,
  so the mesh has glacier and ice-free columns.
