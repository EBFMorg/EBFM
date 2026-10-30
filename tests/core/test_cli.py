# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import unittest
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from ebfm.core.cli import (
    CliDefaults,
    extract_active_coupling_features,
    mpi_launcher_size,
    parse_cli_args,
    validate_forcing_parallel_compat,
    validate_grid_parallel_compat,
    validate_output_frequency_time_step_compat,
    validate_restart_parallel_compat,
    validate_shading_parallel_compat,
    validate_shading_coupling_compat,
    validate_use_part_parallel_compat,
)

from ebfm.core.comm import mpi_available
from ebfm.core.grid import GridInputType
from ebfm.core.config import FieldValidationLevel, Calendar, TimeConfig, iso8601

# Minimal valid args for each primary grid type.
_MATLAB = ["--matlab-mesh", "mesh.mat"]
_ELMER = ["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3413"]
_CARRA2 = ["--netcdf-mesh", "mesh.nc"]
_PARTITIONED_ELMER = _ELMER + ["--netcdf-dem-mesh", "mesh.nc", "--is-partitioned-elmer-mesh"]


def _parse(*extra: str, base: list[str] = _MATLAB):
    """Call parse_cli_args with base grid args plus any extras."""
    return parse_cli_args(list(base) + list(extra))


class TestPrimaryGridMutualExclusion(unittest.TestCase):
    """Exactly one primary grid option must be provided."""

    def test_matlab_mesh_accepted(self):
        args = parse_cli_args(_MATLAB)
        self.assertEqual(args.matlab_mesh, Path("mesh.mat"))

    def test_elmer_mesh_accepted(self):
        args = parse_cli_args(_ELMER)
        self.assertEqual(args.elmer_mesh, Path("mesh/"))

    def test_carra2_mesh_accepted(self):
        args = parse_cli_args(_CARRA2)
        self.assertEqual(args.netcdf_mesh, Path("mesh.nc"))

    def test_no_grid_option_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args([])
        self.assertEqual(ctx.exception.code, 2)

    def test_two_grid_options_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(_MATLAB + _ELMER)
        self.assertEqual(ctx.exception.code, 2)

    def test_two_grid_options_rejected_matlab_carra(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(_MATLAB + _CARRA2)
        self.assertEqual(ctx.exception.code, 2)


class TestFillMissingElevation(unittest.TestCase):
    """--fill-missing-elevation is opt-in and only valid with --netcdf-mesh."""

    def test_off_by_default(self):
        args = parse_cli_args(_CARRA2)
        self.assertFalse(args.fill_missing_elevation)

    def test_accepted_with_netcdf_mesh(self):
        args = parse_cli_args(_CARRA2 + ["--fill-missing-elevation"])
        self.assertTrue(args.fill_missing_elevation)

    def test_rejected_without_netcdf_mesh(self):
        for base in (_MATLAB, _ELMER):
            with self.subTest(base=base[0]):
                with self.assertRaises(SystemExit) as ctx:
                    parse_cli_args(base + ["--fill-missing-elevation"])
                self.assertEqual(ctx.exception.code, 2)


class TestElmerMeshConstraints(unittest.TestCase):
    """Constraints specific to --elmer-mesh."""

    def test_elmer_mesh_without_crs_epsg_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(["--elmer-mesh", "mesh/"])
        self.assertEqual(ctx.exception.code, 2)

    def test_elmer_mesh_crs_epsg_3413_accepted(self):
        args = parse_cli_args(["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3413"])
        self.assertEqual(args.elmer_mesh_crs_epsg, 3413)

    def test_elmer_mesh_crs_epsg_3031_accepted(self):
        args = parse_cli_args(["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3031"])
        self.assertEqual(args.elmer_mesh_crs_epsg, 3031)

    def test_elmer_mesh_crs_epsg_invalid_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "4326"])
        self.assertEqual(ctx.exception.code, 2)


class TestPartitionedMeshConstraints(unittest.TestCase):
    """--is-partitioned-elmer-mesh requires both --elmer-mesh and --netcdf-dem-mesh."""

    def test_partitioned_without_elmer_mesh_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(_MATLAB + ["--is-partitioned-elmer-mesh"])
        self.assertEqual(ctx.exception.code, 2)

    def test_partitioned_without_netcdf_dem_mesh_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3413", "--is-partitioned-elmer-mesh"])
        self.assertEqual(ctx.exception.code, 2)

    @unittest.skipUnless(mpi_available, "requires MPI (pip install 'ebfm[mpi]')")
    def test_partitioned_with_elmer_and_netcdf_dem_accepted(self):
        args = parse_cli_args(
            [
                "--elmer-mesh",
                "mesh/",
                "--elmer-mesh-crs-epsg",
                "3413",
                "--netcdf-dem-mesh",
                "mesh.nc",
                "--is-partitioned-elmer-mesh",
            ]
        )
        self.assertTrue(args.is_partitioned_elmer_mesh)
        self.assertEqual(args.netcdf_dem_mesh, Path("mesh.nc"))


class TestDefaults(unittest.TestCase):
    """Verify default values match CliDefaults."""

    def setUp(self):
        self.args = parse_cli_args(_MATLAB)

    def test_start_time_default(self):
        expected = iso8601(CliDefaults.START_TIME.value)
        self.assertEqual(self.args.start_time, expected)

    def test_end_time_default(self):
        expected = iso8601(CliDefaults.END_TIME.value)
        self.assertEqual(self.args.end_time, expected)

    def test_time_step_default(self):
        self.assertEqual(self.args.time_step, CliDefaults.TIME_STEP_SIZE.value)

    def test_log_level_default(self):
        self.assertEqual(self.args.log_level_console, CliDefaults.LOG_LEVEL_CONSOLE.value)

    def test_component_name_default(self):
        self.assertEqual(self.args.component_name, CliDefaults.COMPONENT_NAME.value)

    def test_local_group_label_defaults_to_component_name(self):
        self.assertEqual(self.args.local_group_label, self.args.component_name)

    def test_field_validation_level_default(self):
        self.assertEqual(self.args.field_validation_level, FieldValidationLevel.FATAL.value)

    def test_with_numba_default_false(self):
        self.assertFalse(self.args.with_numba)

    def test_numba_threads_default_none(self):
        self.assertIsNone(self.args.numba_threads)

    def test_with_gpu_default_false(self):
        self.assertFalse(self.args.with_gpu)

    def test_gpu_vendor_default_auto(self):
        self.assertEqual(self.args.gpu_vendor, "auto")

    def test_no_coupling_by_default(self):
        self.assertFalse(self.args.couple_to_elmer_ice)
        self.assertFalse(self.args.couple_to_icon_atmo)
        self.assertFalse(self.args.couple_to_icon_land)

    def test_netcdf_dem_mesh_default_none(self):
        self.assertIsNone(self.args.netcdf_dem_mesh)

    def test_random_seed_default_none(self):
        self.assertIsNone(self.args.random_seed)

    def test_output_defaults(self):
        self.assertEqual(self.args.output_file, Path("Output") / "model_output.nc")
        self.assertEqual(self.args.output_frequency, timedelta(days=1))


class TestOutputOptions(unittest.TestCase):
    """--output-file / --output-frequency parsing and checks."""

    def test_custom_values_accepted(self):
        args = _parse("--output-file", "/scratch/run1.nc", "--output-frequency", "PT6H")
        self.assertEqual(args.output_file, Path("/scratch/run1.nc"))
        self.assertEqual(args.output_frequency, timedelta(hours=6))

    def test_frequency_other_than_positive_fixed_duration_rejected(self):
        cases = {
            "integer": "8",
            "float": "1.5",
            "word": "daily",
            "zero": "PT0S",
            "negative": "-P1D",
            "months vary in length": "P1M",
            "years vary in length": "P1Y",
        }
        for case, value in cases.items():
            with self.subTest(case, value=value):
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    _parse("--output-frequency", value)

    def test_frequency_must_be_multiple_of_time_step(self):
        cases = {
            "equal to time step": ("PT3H", "PT3H", True),
            "multiple of time step": ("P1D", "PT3H", True),
            "not a multiple": ("PT4H", "PT3H", False),
            "shorter than time step": ("PT1H", "PT3H", False),
        }
        for case, (frequency, time_step, accepted) in cases.items():
            with self.subTest(case, frequency=frequency, time_step=time_step):
                args = _parse("--output-frequency", frequency, "--time-step", time_step)
                time_config = TimeConfig(args)
                if accepted:
                    validate_output_frequency_time_step_compat(args, time_config)
                else:
                    with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                        validate_output_frequency_time_step_compat(args, time_config)


class TestGpuOptions(unittest.TestCase):
    """--with-gpu / --gpu-vendor parsing and compatibility checks."""

    def test_with_gpu_accepted(self):
        args = _parse("--with-gpu")
        self.assertTrue(args.with_gpu)

    def test_gpu_vendor_accepted(self):
        args = _parse("--with-gpu", "--gpu-vendor", "nvidia")
        self.assertEqual(args.gpu_vendor, "nvidia")

    def test_gpu_vendor_invalid_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            _parse("--with-gpu", "--gpu-vendor", "intel")
        self.assertEqual(ctx.exception.code, 2)

    def test_with_gpu_and_with_numba_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            _parse("--with-gpu", "--with-numba")
        self.assertEqual(ctx.exception.code, 2)

    def test_gpu_vendor_without_with_gpu_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            _parse("--gpu-vendor", "nvidia")
        self.assertEqual(ctx.exception.code, 2)


class TestMpiLauncherWithoutMpi4py(unittest.TestCase):
    """Started by an MPI launcher on several processes, EBFM needs mpi4py; otherwise every process runs serially."""

    def _parse(self, environment: dict, mpi_available: bool):
        with patch.dict(os.environ), patch("ebfm.core.cli.mpi_available", mpi_available):
            for variable in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
                os.environ.pop(variable, None)
            os.environ.update(environment)
            self.assertEqual(mpi_launcher_size(), int(next(iter(environment.values()), 1)))
            return parse_cli_args(_MATLAB)

    def test_several_processes_without_mpi4py_rejected(self):
        for environment in ({"OMPI_COMM_WORLD_SIZE": "4"}, {"PMI_SIZE": "2"}):
            with self.subTest(environment=environment), self.assertRaises(SystemExit) as ctx:
                self._parse(environment, mpi_available=False)
            self.assertEqual(ctx.exception.code, 2)

    def test_single_process_without_mpi4py_ok(self):
        for environment in ({}, {"OMPI_COMM_WORLD_SIZE": "1"}):
            with self.subTest(environment=environment):
                self._parse(environment, mpi_available=False)  # must not raise

    def test_several_processes_with_mpi4py_ok(self):
        self._parse({"OMPI_COMM_WORLD_SIZE": "4"}, mpi_available=True)  # must not raise


class TestLocalGroupLabel(unittest.TestCase):
    """--local-group-label overrides the default derived from --component-name."""

    @unittest.skipUnless(mpi_available, "requires MPI (pip install 'ebfm[mpi]')")
    def test_explicit_local_group_label(self):
        args = _parse("--local-group-label", "my-group")
        self.assertEqual(args.local_group_label, "my-group")

    def test_custom_component_name_propagates_to_local_group_label(self):
        args = _parse("--component-name", "ebfm-north")
        self.assertEqual(args.local_group_label, "ebfm-north")


class TestShadingOption(unittest.TestCase):
    """--shading / --no-shading are optional; absence means SUPPRESS (not present in namespace)."""

    def test_shading_not_present_by_default(self):
        args = _parse()
        self.assertFalse(hasattr(args, "shading"))

    def test_shading_true_when_explicit(self):
        args = _parse("--shading")
        self.assertTrue(args.shading)

    def test_shading_false_when_negated(self):
        args = _parse("--no-shading")
        self.assertFalse(args.shading)


class TestCouplingOptions(unittest.TestCase):
    """Coupling flags and their parsing."""

    def test_couple_to_elmer_ice(self):
        args = _parse("--couple-to-elmer-ice")
        self.assertTrue(args.couple_to_elmer_ice)

    def test_couple_to_icon_atmo(self):
        args = _parse("--couple-to-icon-atmo")
        self.assertTrue(args.couple_to_icon_atmo)
        self.assertFalse(args.couple_to_icon_land)

    def test_couple_to_icon_land(self):
        args = _parse("--couple-to-icon-land")
        self.assertTrue(args.couple_to_icon_land)
        self.assertFalse(args.couple_to_icon_atmo)

    def test_couple_to_icon_prefix_is_ambiguous(self):
        # '--couple-to-icon' used to be accepted as an abbreviation of '--couple-to-icon-atmo';
        # with '--couple-to-icon-land' it is ambiguous and must be rejected.
        with self.assertRaises(SystemExit) as ctx:
            _parse("--couple-to-icon")
        self.assertEqual(ctx.exception.code, 2)

    def test_fake_coupling(self):
        args = _parse("--fake-coupling")
        self.assertTrue(args.fake_coupling)

    def test_field_validation_level_warning(self):
        args = _parse("--field-validation-level", "WARNING")
        self.assertEqual(args.field_validation_level, FieldValidationLevel.WARNING.value)

    def test_field_validation_level_silent(self):
        args = _parse("--field-validation-level", "SILENT")
        self.assertEqual(args.field_validation_level, FieldValidationLevel.SILENT.value)

    def test_invalid_field_validation_level_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            _parse("--field-validation-level", "INVALID")
        self.assertEqual(ctx.exception.code, 2)


class TestExtractActiveCouplingFeatures(unittest.TestCase):
    """extract_active_coupling_features returns CLI flag names for active coupling."""

    def test_no_coupling(self):
        args = _parse()
        self.assertEqual(extract_active_coupling_features(args), [])

    def test_couple_to_elmer_ice(self):
        args = _parse("--couple-to-elmer-ice")
        self.assertIn("--couple-to-elmer-ice", extract_active_coupling_features(args))

    def test_couple_to_icon_atmo(self):
        args = _parse("--couple-to-icon-atmo")
        self.assertIn("--couple-to-icon-atmo", extract_active_coupling_features(args))

    def test_couple_to_icon_land(self):
        args = _parse("--couple-to-icon-land")
        self.assertIn("--couple-to-icon-land", extract_active_coupling_features(args))

    def test_coupler_config(self):
        args = _parse("--coupler-config", "coupler.yaml")
        self.assertIn("--coupler-config", extract_active_coupling_features(args))

    def test_multiple_coupling_flags(self):
        args = _parse("--couple-to-elmer-ice", "--couple-to-icon-atmo", "--couple-to-icon-land")
        features = extract_active_coupling_features(args)
        self.assertIn("--couple-to-elmer-ice", features)
        self.assertIn("--couple-to-icon-atmo", features)
        self.assertIn("--couple-to-icon-land", features)


class TestValidateShadingCouplingCompat(unittest.TestCase):
    """validate_shading_coupling_compat raises a CLI error when shading + coupling are both on."""

    def _make_configs(self, use_shading: bool, defines_coupling: bool):
        grid_config = MagicMock()
        grid_config.use_shading = use_shading
        coupling_config = MagicMock()
        coupling_config.defines_coupling.return_value = defines_coupling
        return grid_config, coupling_config

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def test_no_shading_no_coupling_ok(self):
        grid, coupling = self._make_configs(use_shading=False, defines_coupling=False)
        validate_shading_coupling_compat(grid, coupling)  # must not raise

    def test_shading_without_coupling_ok(self):
        grid, coupling = self._make_configs(use_shading=True, defines_coupling=False)
        validate_shading_coupling_compat(grid, coupling)  # must not raise

    def test_coupling_without_shading_ok(self):
        grid, coupling = self._make_configs(use_shading=False, defines_coupling=True)
        validate_shading_coupling_compat(grid, coupling)  # must not raise

    def test_shading_with_coupling_rejected(self):
        grid, coupling = self._make_configs(use_shading=True, defines_coupling=True)
        with self.assertRaises(SystemExit) as ctx:
            validate_shading_coupling_compat(grid, coupling)
        self.assertEqual(ctx.exception.code, 2)


class TestValidateShadingParallelCompat(unittest.TestCase):
    """validate_shading_parallel_compat raises a CLI error when shading is on in a run on several ranks."""

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def _validate(self, use_shading: bool, comm_size: int):
        validate_shading_parallel_compat(MagicMock(use_shading=use_shading), MagicMock(size=comm_size))

    def test_shading_on_single_rank_ok(self):
        self._validate(use_shading=True, comm_size=1)  # must not raise

    def test_no_shading_on_several_ranks_ok(self):
        self._validate(use_shading=False, comm_size=4)  # must not raise

    def test_shading_on_several_ranks_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            self._validate(use_shading=True, comm_size=4)
        self.assertEqual(ctx.exception.code, 2)


class TestValidateGridParallelCompat(unittest.TestCase):
    """validate_grid_parallel_compat rejects every grid but MATLAB and a partitioned Elmer mesh on several ranks."""

    # name: (grid_type, is_partitioned, grid option named by the error on several ranks, or None if accepted)
    _CASES = {
        "--matlab-mesh": (GridInputType.MATLAB, False, None),
        "--netcdf-mesh": (GridInputType.NETCDF, False, "--netcdf-mesh"),
        "--elmer-mesh": (GridInputType.ELMER, False, "--elmer-mesh"),
        "--elmer-mesh --netcdf-dem-mesh": (GridInputType.CUSTOM, False, "--elmer-mesh"),
        "--elmer-mesh --netcdf-dem-mesh-unstructured": (GridInputType.ELMERXIOS, False, "--elmer-mesh"),
        "--elmer-mesh --netcdf-dem-mesh --is-partitioned-elmer-mesh": (GridInputType.CUSTOM, True, None),
    }

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def _validate(self, grid_type, is_partitioned: bool, comm_size: int):
        grid_config = MagicMock(grid_type=grid_type, is_partitioned=is_partitioned)
        validate_grid_parallel_compat(grid_config, MagicMock(size=comm_size))

    def test_single_rank_ok(self):
        for name, (grid_type, is_partitioned, _) in self._CASES.items():
            with self.subTest(name):
                self._validate(grid_type, is_partitioned, comm_size=1)  # must not raise

    def test_several_ranks(self):
        for name, (grid_type, is_partitioned, rejected_grid) in self._CASES.items():
            with self.subTest(name):
                if rejected_grid is None:
                    self._validate(grid_type, is_partitioned, comm_size=4)  # must not raise
                else:
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                        self._validate(grid_type, is_partitioned, comm_size=4)
                    self.assertEqual(ctx.exception.code, 2)
                    error = stderr.getvalue().splitlines()[-1]
                    self.assertIn(f"error: {rejected_grid} ", error)
                    self.assertIn("Only --matlab-mesh and a partitioned Elmer mesh", error)


@unittest.skipUnless(mpi_available, "requires MPI (pip install 'ebfm[mpi]')")
class TestValidateUsePartParallelCompat(unittest.TestCase):
    """validate_use_part_parallel_compat raises a CLI error for --use-part in a run on several ranks."""

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def _validate(self, use_part, comm_size: int):
        args = _parse(*(["--use-part", use_part] if use_part else []), base=_PARTITIONED_ELMER)
        validate_use_part_parallel_compat(args, MagicMock(size=comm_size))

    def test_use_part_on_single_rank_ok(self):
        self._validate(use_part="42", comm_size=1)  # must not raise

    def test_no_use_part_on_several_ranks_ok(self):
        self._validate(use_part=None, comm_size=4)  # must not raise

    def test_use_part_on_several_ranks_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            self._validate(use_part="42", comm_size=4)
        self.assertEqual(ctx.exception.code, 2)


class TestValidateForcingParallelCompat(unittest.TestCase):
    """validate_forcing_parallel_compat raises a CLI error when forcing files are read in a run on several ranks."""

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def _validate(self, forcing_dir, comm_size: int):
        args = _parse(*(["--forcing-dir", forcing_dir] if forcing_dir else []))
        validate_forcing_parallel_compat(args, MagicMock(size=comm_size))

    def test_forcing_on_single_rank_ok(self):
        self._validate(forcing_dir="forcing/", comm_size=1)  # must not raise

    def test_no_forcing_on_several_ranks_ok(self):
        self._validate(forcing_dir=None, comm_size=4)  # must not raise

    def test_forcing_on_several_ranks_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            self._validate(forcing_dir="forcing/", comm_size=4)
        self.assertEqual(ctx.exception.code, 2)


class TestValidateRestartParallelCompat(unittest.TestCase):
    """validate_restart_parallel_compat raises a CLI error when restart files are used in a run on several ranks."""

    _RESTART_ARGS = {
        "--restart-dir": ["--restart-dir", "restart/"],
        "--restart-init": ["--restart-init"],
        "--restart-dir --restart-init": ["--restart-dir", "restart/", "--restart-init"],
    }

    def setUp(self):
        # Ensure _parser is set by running a valid parse first.
        parse_cli_args(_MATLAB)

    def _validate(self, restart_args, comm_size: int, base: list[str] = _MATLAB):
        validate_restart_parallel_compat(_parse(*restart_args, base=base), MagicMock(size=comm_size))

    def test_restart_on_single_rank_ok(self):
        for name, restart_args in self._RESTART_ARGS.items():
            with self.subTest(name):
                self._validate(restart_args, comm_size=1)  # must not raise

    def test_no_restart_on_several_ranks_ok(self):
        self._validate([], comm_size=4)  # must not raise

    def test_restart_on_several_ranks_rejected(self):
        for name, restart_args in self._RESTART_ARGS.items():
            with self.subTest(name), self.assertRaises(SystemExit) as ctx:
                self._validate(restart_args, comm_size=4)
            self.assertEqual(ctx.exception.code, 2)

    @unittest.skipUnless(mpi_available, "requires MPI (pip install 'ebfm[mpi]')")
    def test_restart_on_several_ranks_rejected_for_partitioned_mesh(self):
        """A partitioned mesh runs on several ranks, but without restart files."""
        for name, restart_args in self._RESTART_ARGS.items():
            with self.subTest(name), self.assertRaises(SystemExit) as ctx:
                self._validate(restart_args, comm_size=4, base=_PARTITIONED_ELMER)
            self.assertEqual(ctx.exception.code, 2)


class TestCliCalendarArgument(unittest.TestCase):
    def test_calendar_default(self):
        args = parse_cli_args(_MATLAB)
        self.assertEqual(args.calendar, CliDefaults.CALENDAR.value)

    def test_calendar_accepts_all_supported_values(self):
        for calendar in Calendar:
            args = parse_cli_args(_MATLAB + ["--calendar", calendar.value])
            self.assertEqual(args.calendar, calendar.value)

    def test_calendar_rejects_invalid_value(self):
        with self.assertRaises(SystemExit):
            parse_cli_args(_MATLAB + ["--calendar", "invalid_calendar"])


class TestCliHelpOutput(unittest.TestCase):
    def test_help_lists_full_parser_options(self):
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(buf):
                parse_cli_args(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn("--matlab-mesh", help_text)
        self.assertIn("--elmer-mesh", help_text)
        self.assertIn("--netcdf-mesh", help_text)
        self.assertIn("--calendar", help_text)


class TestTimeConfigCalendar(unittest.TestCase):
    def test_time_config_uses_calendar_from_cli_args(self):
        args = parse_cli_args(_MATLAB + ["--calendar", Calendar.YEAR_OF_365_DAYS.value])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.calendar, Calendar.YEAR_OF_365_DAYS)


class TestTimeConfigStartEndTimes(unittest.TestCase):
    start_time_str = "01-Jan-2020 00:00"
    end_time_str = "31-Dec-2020 23:59"

    start_time_iso = "2020-01-01T00:00:00"
    end_time_iso = "2020-12-31T23:59:59"

    def test_time_config_parses_start_end_times_from_cli_args(self):
        args = parse_cli_args(_MATLAB + ["--start-time", self.start_time_str, "--end-time", self.end_time_str])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.start_time.strftime("%d-%b-%Y %H:%M"), self.start_time_str)
        self.assertEqual(time_config.end_time.strftime("%d-%b-%Y %H:%M"), self.end_time_str)

    def test_time_config_parses_start_end_times_from_cli_args_iso8601(self):

        args = parse_cli_args(_MATLAB + ["--start-time", self.start_time_iso, "--end-time", self.end_time_iso])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.start_time.strftime("%Y-%m-%dT%H:%M:%S"), self.start_time_iso)
        self.assertEqual(time_config.end_time.strftime("%Y-%m-%dT%H:%M:%S"), self.end_time_iso)


class TestTimeConfigTimeStep(unittest.TestCase):
    time_step_iso_in = "P2DT12H"  # 2 days and 12 hours
    time_step_iso_out = "P2DT12H0M0S"  # 2 days and 12 hours
    time_step_days = 2.5

    def test_time_config_parses_time_step_from_cli_args(self):
        args = parse_cli_args(_MATLAB + ["--time-step", str(self.time_step_days)])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.time_step_in_days(), self.time_step_days)
        self.assertEqual(time_config.time_step_iso8601(), self.time_step_iso_out)

    def test_time_config_parses_time_step_from_cli_args_iso8601(self):
        args = parse_cli_args(_MATLAB + ["--time-step", self.time_step_iso_in])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.time_step_in_days(), self.time_step_days)
        self.assertEqual(time_config.time_step_iso8601(), self.time_step_iso_out)


if __name__ == "__main__":
    unittest.main()
