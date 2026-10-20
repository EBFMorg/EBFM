# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import netCDF4
import numpy as np

import ebfm.reader
from ebfm.reader import read_dem

NX, NY = 40, 30


def _write_dem(dem_file: Path) -> None:
    """Write a DEM on a 1 km grid with a descending y axis, like BedMachine, whose surface is unique per point."""
    with netCDF4.Dataset(dem_file, "w") as nc:
        nc.createDimension("x", NX)
        nc.createDimension("y", NY)
        nc.createVariable("x", "f8", ("x",))[:] = 1000.0 * np.arange(NX)
        nc.createVariable("y", "f8", ("y",))[:] = -1000.0 * np.arange(NY)
        nc.createVariable("surface", "f4", ("y", "x"))[:] = np.arange(NX * NY, dtype=np.float32).reshape(NY, NX)


class TestReadDem(unittest.TestCase):
    """Sampling a DEM gives the surface at the nearest grid point, whichever bands the DEM is read in."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dem_file = Path(tmp.name) / "dem.nc"
        _write_dem(self.dem_file)

        rng = np.random.default_rng(0)
        self.ix = rng.integers(0, NX, size=200)
        self.iy = rng.integers(0, NY, size=200)
        # Off the grid points by less than half a spacing
        self.xs = 1000.0 * self.ix + rng.uniform(-400, 400, size=200)
        self.ys = -1000.0 * self.iy + rng.uniform(-400, 400, size=200)

    def test_samples_nearest_grid_point(self):
        band_sizes = {"whole DEM": NX * NY, "several rows": 3 * NX, "one row": NX, "less than a row": 1}
        expected = (NX * self.iy + self.ix).astype(np.float32)
        for name, band_size in band_sizes.items():
            with self.subTest(name), mock.patch.object(ebfm.reader, "DEM_BAND_SIZE", band_size):
                result = read_dem(self.dem_file, self.xs, self.ys)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
