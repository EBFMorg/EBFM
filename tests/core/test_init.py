# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from datetime import datetime, timezone

import numpy as np
from netCDF4 import Dataset

from ebfm.core import INIT
from ebfm.core.INIT import init_config, init_grid, init_initial_conditions
from ebfm.core.FINAL_create_restart_file import main as write_restart_file
from ebfm.core.cli import parse_cli_args
from ebfm.core.config import ColumnDiscretizationConfig
from ebfm.core.config.grid import GridConfig
from ebfm.core.config.time import TimeConfig
from ebfm.core.restart import PER_COLUMN_VARIABLES, PER_LAYER_VARIABLES, RESTART_VARIABLES

GPSUM = 5
NL = 4

_MATLAB_MESH = Path(__file__).parents[2] / "examples" / "dem_and_mask.mat"


def _make_column():
    """A discretization matching the restart files written by these tests."""
    return ColumnDiscretizationConfig(nl=NL, split=(2,))


def _make_grid(number_of_columns: int = GPSUM) -> dict:
    """A minimal grid carrying `number_of_columns` columns.

    `init_initial_conditions` derives the column count from the grid via
    `ebfm.core.grid.number_of_columns`, so the mask is what sets it.
    """
    return {"mask": np.ones(number_of_columns, dtype=int)}


def _build_out_dict() -> dict:
    return {
        "subZ": np.full((GPSUM, NL), 1.0),
        "subW": np.full((GPSUM, NL), 1.0),
        "subD": np.full((GPSUM, NL), 1.0),
        "subS": np.full((GPSUM, NL), 1.0),
        "subT": np.full((GPSUM, NL), 1.0),
        "subTmean": np.full((GPSUM, NL), 1.0),
        "snowmass": np.full((GPSUM,), 1.0),
        "Tsurf": np.full((GPSUM,), 1.0),
        "ys": np.full((GPSUM,), 1.0),
        "timelastsnow": np.array([datetime(1979, 1, 1, tzinfo=timezone.utc)] * GPSUM),
        "alb_snow": np.full((GPSUM,), 1.0),
        "surface_elevation": np.full((GPSUM,), 1.0),
        "x": np.arange(GPSUM, dtype=np.int32),
        "y": np.arange(GPSUM, dtype=np.int32),
    }


def _write_restart_file(path: Path, with_missing_value: bool = False) -> None:
    io = {"writebootfile": True, "bootfileout": path}
    write_restart_file(_build_out_dict(), io, restartdir=path.parent)

    if with_missing_value:
        # Restart files produced by EBFM are always fully populated. To
        # exercise the "contains missing values" guard, reopen the file
        # afterwards and simulate corruption/an incomplete write.
        with Dataset(path, "r+") as ncfile:
            ncfile.variables["subZ"][0, 0] = np.ma.masked


class TestInitFromRestartFile(unittest.TestCase):
    """
    Regression test for a bug where restart-loaded arrays stayed
    `numpy.ma.MaskedArray` (netCDF4's default) instead of plain `ndarray`.

    Mixing masked arrays into the simulation state made numpy's masked-array
    ufuncs silently ignore `where=` when combining masks in LOOP_SNOW's
    compaction() step, so masks from expected divide-by-zero results leaked
    into cells they should not and snowballed across timesteps, eventually
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
        grid = _make_grid()
        io = {"bootfilein": self.bootfile}
        time = {}

        OUT, _, _ = init_initial_conditions(C, grid, io, time, _make_column(), init_with_restart_file=True)

        for var_name in ("subZ", "subW", "subD", "subS", "subT", "subTmean", "Tsurf", "ys", "alb_snow"):
            with self.subTest(var_name=var_name):
                self.assertIsInstance(OUT[var_name], np.ndarray)
                self.assertNotIsInstance(OUT[var_name], np.ma.MaskedArray)

    def test_restart_arrays_keep_their_values(self):
        C = {"alb_ice": 0.5}
        grid = _make_grid()
        io = {"bootfilein": self.bootfile}
        time = {}

        OUT, _, _ = init_initial_conditions(C, grid, io, time, _make_column(), init_with_restart_file=True)

        np.testing.assert_array_equal(OUT["subZ"], np.full((GPSUM, NL), 1.0))
        np.testing.assert_array_equal(OUT["subD"], np.full((GPSUM, NL), 1.0))

    def test_restart_file_with_missing_values_is_rejected(self):
        bootfile_with_gap = Path(self.tmpdir.name) / "restart_test_missing.nc"
        _write_restart_file(bootfile_with_gap, with_missing_value=True)

        C = {"alb_ice": 0.5}
        grid = _make_grid()
        io = {"bootfilein": bootfile_with_gap}
        time = {}

        with self.assertRaises(AssertionError):
            init_initial_conditions(C, grid, io, time, _make_column(), init_with_restart_file=True)


class TestRestartFileShapeValidation(unittest.TestCase):
    """
    A restart file must match the configured column discretization.

    The restart branch of `init_initial_conditions` takes its array shapes from the file, while the
    tail shared with the manual branch allocates `subK`, `subCeff` and `subWvol` at the configured
    `(gpsum, nl)`. Without this check a mismatched file leaves `OUT` holding two different column or
    layer counts at once, and nothing downstream reports it usefully: a wrong column count fails on
    the first timestep with an `IndexError` naming neither the file nor the cause, and a wrong layer
    count runs the whole simulation before failing in the output writer -- or completes with no error
    at all when output writing never triggers, writing a fresh restart file as if nothing were wrong.

    The matching case is covered by `TestInitFromRestartFile`, which loads the same files with the
    same grid and column config; those tests fail if this validation is too strict.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.bootfile = Path(self.tmpdir.name) / "restart_test.nc"
        _write_restart_file(self.bootfile)
        self.C = {"alb_ice": 0.5}
        self.io = {"bootfilein": self.bootfile}
        self.time = {}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_restart_file_with_wrong_layer_count_is_rejected(self):
        for nl in (NL - 1, NL + 1):
            with self.subTest(nl=nl):
                column = ColumnDiscretizationConfig(nl=nl, split=(2,))
                with self.assertRaisesRegex(ValueError, r"'subZ'.*must have shape"):
                    init_initial_conditions(
                        self.C, _make_grid(), self.io, self.time, column, init_with_restart_file=True
                    )

    def test_restart_file_with_wrong_column_count_is_rejected(self):
        # A single column is included as the extreme case: it is the one where the per-layer arrays
        # would broadcast against the configured `(gpsum, nl)` rather than fail to align.
        for number_of_columns in (1, GPSUM - 1, GPSUM + 1):
            with self.subTest(number_of_columns=number_of_columns):
                with self.assertRaisesRegex(ValueError, r"'subZ'.*must have shape"):
                    init_initial_conditions(
                        self.C,
                        _make_grid(number_of_columns),
                        self.io,
                        self.time,
                        _make_column(),
                        init_with_restart_file=True,
                    )

    def test_restart_file_with_wrong_length_per_column_variable_is_rejected(self):
        # Every per-layer variable matches here, so only the per-column check can catch this.
        bootfile_short = Path(self.tmpdir.name) / "restart_test_short.nc"
        _write_restart_file(bootfile_short)
        with Dataset(bootfile_short, "r+") as ncfile:
            ncfile.createDimension("short", GPSUM - 2)
            ncfile.createVariable("ys_short", "f8", ("short",))[:] = 1.0

        with self.assertRaisesRegex(ValueError, r"'ys_short'.*per-column variable must have shape"):
            init_initial_conditions(
                self.C,
                _make_grid(),
                {"bootfilein": bootfile_short},
                self.time,
                _make_column(),
                init_with_restart_file=True,
            )

    def _load(self, bootfile):
        return init_initial_conditions(
            self.C, _make_grid(), {"bootfilein": bootfile}, self.time, _make_column(), init_with_restart_file=True
        )

    def _write_with(self, name: str, **overrides) -> Path:
        """Write a restart file whose contents differ from a valid one by `overrides`."""
        bootfile = Path(self.tmpdir.name) / f"restart_test_{name}.nc"
        out = _build_out_dict()
        out.update(overrides)
        write_restart_file(out, {"writebootfile": True, "bootfileout": bootfile}, restartdir=bootfile.parent)
        return bootfile

    def test_per_layer_variable_stored_per_column_is_rejected(self):
        # `(GPSUM,)` is a perfectly legal restart shape -- it is what the per-column variables use --
        # so only knowing that `subZ` belongs to the per-layer group can catch this.
        bootfile = self._write_with("sub_z_1d", subZ=np.full((GPSUM,), 1.0))

        with self.assertRaisesRegex(ValueError, r"'subZ'.*must have shape"):
            self._load(bootfile)

    def test_per_column_variable_stored_per_layer_is_rejected(self):
        # The mirror image: `(GPSUM, NL)` is legal for a per-layer variable, wrong for `Tsurf`.
        bootfile = self._write_with("tsurf_2d", Tsurf=np.full((GPSUM, NL), 1.0))

        with self.assertRaisesRegex(ValueError, r"'Tsurf'.*must have shape"):
            self._load(bootfile)

    def test_restart_file_missing_a_required_variable_is_rejected(self):
        # Copy a valid file, dropping `subZ`. Without the manifest check the model only notices when
        # the time loop indexes `OUT["subZ"]`, far from the cause.
        bootfile = Path(self.tmpdir.name) / "restart_test_no_sub_z.nc"
        with Dataset(self.bootfile) as src, Dataset(bootfile, "w", format="NETCDF4") as dst:
            for dim_name, dimension in src.dimensions.items():
                dst.createDimension(dim_name, len(dimension))
            for var_name, variable in src.variables.items():
                if var_name == "subZ":
                    continue
                dst.createVariable(var_name, variable.dtype, variable.dimensions)[:] = variable[:]

        with self.assertRaisesRegex(ValueError, r"missing the required variable\(s\): subZ"):
            self._load(bootfile)

    def test_variable_outside_the_manifest_is_accepted_at_a_valid_shape(self):
        # Auxiliary variables may be added to a restart file without touching the loader, as long as
        # they carry a recognisable column axis.
        bootfile = Path(self.tmpdir.name) / "restart_test_extra.nc"
        _write_restart_file(bootfile)
        with Dataset(bootfile, "r+") as ncfile:
            ncfile.createVariable("an_extra_diagnostic", "f8", ("ys_dim0",))[:] = 2.0

        OUT, _, _ = self._load(bootfile)

        np.testing.assert_array_equal(OUT["an_extra_diagnostic"], np.full((GPSUM,), 2.0))

    def test_restart_file_with_too_many_dimensions_is_rejected(self):
        bootfile_3d = Path(self.tmpdir.name) / "restart_test_3d.nc"
        _write_restart_file(bootfile_3d)
        with Dataset(bootfile_3d, "r+") as ncfile:
            # Correct column and layer count, but one dimension too many.
            ncfile.createDimension("extra", 2)
            variable = ncfile.createVariable("subZ_3d", "f8", ("subZ_dim0", "subZ_dim1", "extra"))
            variable[:] = 1.0

        with self.assertRaisesRegex(ValueError, r"'subZ_3d' .*has 3 dimensions"):
            init_initial_conditions(
                self.C,
                _make_grid(),
                {"bootfilein": bootfile_3d},
                self.time,
                _make_column(),
                init_with_restart_file=True,
            )

    def test_restart_file_with_a_scalar_variable_is_accepted(self):
        # The writer stores scalars as 0-d variables, which carry no column or layer axis to check.
        bootfile_scalar = Path(self.tmpdir.name) / "restart_test_scalar.nc"
        _write_restart_file(bootfile_scalar)
        with Dataset(bootfile_scalar, "r+") as ncfile:
            ncfile.createVariable("a_scalar", "f8", ()).assignValue(1.0)

        OUT, _, _ = init_initial_conditions(
            self.C,
            _make_grid(),
            {"bootfilein": bootfile_scalar},
            self.time,
            _make_column(),
            init_with_restart_file=True,
        )

        self.assertEqual(OUT["a_scalar"], 1.0)


class TestRestartVariableManifest(unittest.TestCase):
    """The writer and the loader must agree on what a restart file contains.

    Both derive it from `ebfm.core.restart.RESTART_VARIABLES`; these tests pin that they cannot
    drift apart, which is the whole point of sharing the manifest.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writer_writes_exactly_the_manifest(self):
        bootfile = Path(self.tmpdir.name) / "restart_test.nc"
        _write_restart_file(bootfile)

        with Dataset(bootfile) as ncfile:
            self.assertEqual(set(ncfile.variables), set(RESTART_VARIABLES))

    def test_writer_writes_each_group_at_its_own_shape(self):
        bootfile = Path(self.tmpdir.name) / "restart_test.nc"
        _write_restart_file(bootfile)

        with Dataset(bootfile) as ncfile:
            for var_name in PER_LAYER_VARIABLES:
                with self.subTest(var_name=var_name):
                    self.assertEqual(ncfile.variables[var_name].shape, (GPSUM, NL))
            for var_name in PER_COLUMN_VARIABLES:
                with self.subTest(var_name=var_name):
                    self.assertEqual(ncfile.variables[var_name].shape, (GPSUM,))

    def test_the_two_groups_do_not_overlap(self):
        self.assertEqual(set(PER_LAYER_VARIABLES) & set(PER_COLUMN_VARIABLES), set())
        self.assertEqual(len(RESTART_VARIABLES), len(set(RESTART_VARIABLES)))


class TestMatlabShadingLookupTable(unittest.TestCase):
    """The shading look-up table must only be pre-computed when shading is enabled.

    Building it is by far the most expensive part of grid initialization (ca. 25 s for
    examples/dem_and_mask.mat), so the test never lets it run: the disabled case skips
    it by design and the enabled case patches the helper and only checks that it is
    called. What the look-up table contains is not affected by this.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _init_grid(self, shading_flag):
        args = parse_cli_args(["--matlab-mesh", str(_MATLAB_MESH), shading_flag])
        grid_config = GridConfig(args)
        grid, io, _, _ = init_config(TimeConfig(args), grid_config, Path(self.tmpdir.name), False)
        return init_grid(grid, io, grid_config)

    def test_shading_disabled_skips_the_lookup_table(self):
        grid = self._init_grid("--no-shading")

        self.assertFalse(grid["has_shading"])
        for key in ("maxgridangle", "az_array", "nr_az_steps", "shading_method"):
            self.assertNotIn(key, grid, f"{key} must not be created when shading is disabled")

    def test_shading_enabled_precomputes_the_lookup_table(self):
        # Swap the helper for a recorder while init_grid runs: this checks that the
        # call happens, without building the (ca. 25 s) look-up table itself.
        with mock.patch.object(INIT, "_precompute_shading_matlab") as precompute:
            grid = self._init_grid("--shading")

        self.assertTrue(grid["has_shading"])
        precompute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
