# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import netCDF4
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

nc = netCDF4.Dataset("model_output.nc")

xv = nc["x_vertex"][:]
yv = nc["y_vertex"][:]
triangles = nc["cell_vertices"][:]
smb = nc["smb"][0, :]

triangulation = mtri.Triangulation(xv, yv, triangles)

plt.tripcolor(triangulation, facecolors=smb, shading="flat")
plt.colorbar(label="smb")
plt.gca().set_aspect("equal")
plt.show()
