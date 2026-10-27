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

`grid.nc` and the meteorological forcing files were produced using the CARRA2 reanalysis dataset for Greenland. 
The raw data, downloaded from the CARRA2 web portal (https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra-means?tab=download), were vectorized such that only grid cells covering the Greenland Ice Sheet were kept. 
To check the included meteorological fields, use ncdump on the forcing files (e.g., `ncdump -h examples/Greenland_CARRA2/forcing/1990-05-01/P1M.nc`).

Reference:
Copernicus Climate Change Service, Climate Data Store, (2025): pan-Arctic regional reanalysis daily and monthly data from 1986 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/b19380d8 (Accessed on DD-MMM-YYYY)


