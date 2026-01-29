<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# develop

* Fix and extend `reader.py`, documentation on how to use it and how to obtain required example data. https://github.com/EBFMorg/EBFM/pull/69.
* Revise folder layout to avoid clutter in `site-packages`. Installing EBFM should now only affect `site-packages/ebfm`. https://github.com/EBFMorg/EBFM/pull/73.
* Require Python minimum version 3.9. (Planned to increase to 3.10 soon)
* Clarification how `--start-time` and `--end-time` is interpreted by EBFM. Require that difference of start and end time is a multiple of `--time-step`. https://github.com/EBFMorg/EBFM/pull/58.
* Allow logger configuration via command-line interface. Refer to `ebfm --help` and the options `--log-level-console` and `--log-file`. See https://github.com/EBFMorg/EBFM/pull/56.
* Support new input mesh format. EBFM now accepts Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS. To use this feature please provide `--elmer-mesh` together with the new option `--netcdf-mesh-unstructured`. See https://github.com/EBFMorg/EBFM/pull/12.
* Use `setuptools_scm` as backend for `--version` information. See https://github.com/EBFMorg/EBFM/pull/46.
* Remove `pathlib` from requirements, because this can lead to a bug. https://github.com/EBFMorg/EBFM/pull/48.

# v0.1.0

* Initial release
