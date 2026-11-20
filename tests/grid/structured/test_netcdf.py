# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ebfm.grid.structured.netcdf import read_NETCDF_grid


class TestReadNetcdfGrid(unittest.TestCase):
    """Missing orography is NaN unless the user opts in to filling it with zero elevation.

    Filling must only touch cells without orography data. Inactive cells with valid orography (e.g. ice-free land
    next to the ice margin) must keep their elevation, otherwise `np.gradient` sees an artificial cliff and
    over-estimates the slope of the adjacent active cells.
    """

    _FILL = -9999.0

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        # Row 0: active, active, missing orography (but mask > 0)
        # Row 1: active, inactive with valid orography, missing orography (mask == 0)
        self.gridfile = self._write_grid("grid.nc", [[1000.0, 1100.0, self._FILL], [900.0, 500.0, self._FILL]])

    def _write_grid(self, name: str, orog) -> Path:
        gridfile = Path(self.tmpdir.name) / name
        mask = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        lon, lat = np.meshgrid([-45.0, -44.9, -44.8], [72.0, 72.1])

        with Dataset(gridfile, "w") as nc:
            nc.createDimension("y", 2)
            nc.createDimension("x", 3)
            nc.createVariable("orog", "f8", ("y", "x"), fill_value=self._FILL)[:] = np.array(orog)
            nc.createVariable("mask", "f8", ("y", "x"))[:] = mask
            nc.createVariable("latitude", "f8", ("y", "x"))[:] = lat
            nc.createVariable("longitude", "f8", ("y", "x"))[:] = lon
        return gridfile

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_missing_orography_is_inactive(self):
        for fill in (False, True):
            with self.subTest(fill_missing_elevation=fill):
                data = read_NETCDF_grid(self.gridfile, fill_missing_elevation=fill)
                np.testing.assert_array_equal(data["mask"], [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    def test_missing_orography_is_nan_by_default(self):
        data = read_NETCDF_grid(self.gridfile)
        self.assertTrue(np.isnan(data["z"][0, 2]))
        self.assertTrue(np.isnan(data["z"][1, 2]))

    def test_missing_orography_is_filled_with_zero_on_request(self):
        data = read_NETCDF_grid(self.gridfile, fill_missing_elevation=True)
        self.assertEqual(data["z"][0, 2], 0.0)
        self.assertEqual(data["z"][1, 2], 0.0)
        self.assertTrue(np.isfinite(data["z"]).all())

    def test_filling_is_logged(self):
        with self.assertLogs("ebfm.grid.structured.netcdf", level="WARNING") as logs:
            read_NETCDF_grid(self.gridfile, fill_missing_elevation=True)
        self.assertIn("Setting 2 missing elevations", logs.output[0])

    def test_nothing_logged_without_missing_orography(self):
        gridfile = self._write_grid("complete.nc", [[1000.0, 1100.0, 1200.0], [900.0, 500.0, 400.0]])
        with self.assertNoLogs("ebfm.grid.structured.netcdf", level="WARNING"):
            read_NETCDF_grid(gridfile, fill_missing_elevation=True)

    def test_valid_orography_of_inactive_cell_is_kept(self):
        for fill in (False, True):
            with self.subTest(fill_missing_elevation=fill):
                data = read_NETCDF_grid(self.gridfile, fill_missing_elevation=fill)
                self.assertEqual(data["z"][1, 1], 500.0)
