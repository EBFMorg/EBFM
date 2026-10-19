<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Greenland CARRA2 case

This folder contains

* `grid.nc`: a grid file of Greenland
* `forcing`: a folder with forcing files.
* `forcing/1990-05-01/P1M.nc`: the forcing file covering one month (ISO8601 `P1M`) starting from `1990-05-01`.

The path of a forcing file follows the ISO8601 notation for a time interval with start date and duration, for example `1990-05-01/P1M`.
EBFM reads one file per month from `<forcing-dir>/<YYYY-MM-01>/P1M.nc`, so further months can be added as additional folders.
Each forcing file must provide its data on the active cells of the grid it is used with; a grid with a different resolution therefore needs its own forcing folder.

## How the files were produced

### Grid file `grid.nc`

TODO

### Forcing files

TODO
