<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Example Files

| File                           | Description                                    |
|------------------------------- | ---------------------------------------------- |
| `BedMachineGreenland-v5_lo.nc` | Low-res version of `BedMachineGreenland-v5.nc` |
| `dem_and_mask.mat`             | MATLAB example grid                            |

### How `BedMachineGreenland-v5_lo.nc` was produced

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
