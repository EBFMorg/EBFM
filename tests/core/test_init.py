# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import unittest
import tempfile
from pathlib import Path

from datetime import datetime, timezone

import numpy as np
from netCDF4 import Dataset, date2num

from ebfm.core.INIT import init_initial_conditions

GPSUM = 5
NL = 4


def _write_restart_file(path: Path, with_missing_value: bool = False) -> None:
    with Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("gpsum", GPSUM)
        ncfile.createDimension("nl", NL)

        for var_name in ("subZ", "subW", "subD", "subS", "subT", "subTmean"):
            var = ncfile.createVariable(var_name, "f8", ("gpsum", "nl"))
            var[:, :] = np.full((GPSUM, NL), 1.0)

        for var_name in ("snowmass", "Tsurf", "ys", "alb_snow", "surface_elevation"):
            var = ncfile.createVariable(var_name, "f8", ("gpsum",))
            var[:] = np.full((GPSUM,), 1.0)

        for var_name in ("x", "y"):
            var = ncfile.createVariable(var_name, "i4", ("gpsum",))
            var[:] = np.arange(GPSUM, dtype="i4")

        timelastsnow = ncfile.createVariable("timelastsnow_netCDF", "f8", ("gpsum",))
        timelastsnow[:] = date2num(
            [datetime(1979, 1, 1, tzinfo=timezone.utc)] * GPSUM,
            units="days since 1970-01-01 00:00:00",
            calendar="gregorian",
        )

        if with_missing_value:
            # Leave one entry of subZ unwritten so netCDF4 reports it as masked
            # (missing) when read back with auto-masking enabled (the default).
            ncfile.variables["subZ"][0, 0] = np.ma.masked


class TestInitFromRestartFile(unittest.TestCase):
    """
    Regression test for a bug where restart-loaded arrays stayed
    `numpy.ma.MaskedArray` (netCDF4's default) instead of plain `ndarray`.

    Mixing masked arrays into the simulation state made numpy's masked-array
    ufuncs silently ignore `where=` when combining masks in LOOP_SNOW's
    compaction() step, so masks from expected divide-by-zero results leaked
    into cells they shouldn't and snowballed across timesteps, eventually
    making `dt_stab` fully masked and tripping
    `assert (dt_stab > 0).all()` even though no real value was <= 0.

    See https://github.com/EBFMorg/EBFM/pull/154 for the fix and discussion.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.bootfile = Path(self.tmpdir.name) / "restart_test.nc"
        _write_restart_file(self.bootfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_restart_arrays_are_not_masked(self):
        C = {"alb_ice": 0.5}
        grid = {"gpsum": GPSUM, "nl": NL}
        io = {"bootfilein": self.bootfile}
        time = {}

        OUT, IN, OUTFILE = init_initial_conditions(C, grid, io, time, init_with_restart_file=True)

        for var_name in ("subZ", "subW", "subD", "subS", "subT", "subTmean", "Tsurf", "ys", "alb_snow"):
            with self.subTest(var_name=var_name):
                self.assertIsInstance(OUT[var_name], np.ndarray)
                self.assertNotIsInstance(OUT[var_name], np.ma.MaskedArray)

    def test_restart_arrays_keep_their_values(self):
        C = {"alb_ice": 0.5}
        grid = {"gpsum": GPSUM, "nl": NL}
        io = {"bootfilein": self.bootfile}
        time = {}

        OUT, IN, OUTFILE = init_initial_conditions(C, grid, io, time, init_with_restart_file=True)

        np.testing.assert_array_equal(OUT["subZ"], np.full((GPSUM, NL), 1.0))
        np.testing.assert_array_equal(OUT["subD"], np.full((GPSUM, NL), 1.0))

    def test_restart_file_with_missing_values_is_rejected(self):
        bootfile_with_gap = Path(self.tmpdir.name) / "restart_test_missing.nc"
        _write_restart_file(bootfile_with_gap, with_missing_value=True)

        C = {"alb_ice": 0.5}
        grid = {"gpsum": GPSUM, "nl": NL}
        io = {"bootfilein": bootfile_with_gap}
        time = {}

        with self.assertRaises(AssertionError):
            init_initial_conditions(C, grid, io, time, init_with_restart_file=True)


if __name__ == "__main__":
    unittest.main()
