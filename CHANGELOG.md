<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# develop

* Support new input mesh format. EBFM now accepts Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS. To use this feature please provide `--elmer-mesh` together with the new option `--netcdf-mesh-unstructured`. See https://github.com/EBFMorg/EBFM/pull/12.
* Use `setuptools_scm` as backend for `--version` information. See https://github.com/EBFMorg/EBFM/pull/46.
* Remove `pathlib` from requirements, because this can lead to a bug. https://github.com/EBFMorg/EBFM/pull/48.

# v0.1.0

* Initial release
