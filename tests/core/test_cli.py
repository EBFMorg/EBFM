# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

from ebfm.core.cli import (
    CliDefaults,
    extract_active_coupling_features,
    parse_cli_args,
    validate_shading_coupling_compat,
)

from ebfm.core.config import FieldValidationLevel, Calendar, TimeConfig

# Minimal valid args for each primary grid type.
_MATLAB = ["--matlab-mesh", "mesh.mat"]
_ELMER = ["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3413"]


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

    def test_no_grid_option_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args([])
        self.assertEqual(ctx.exception.code, 2)

    def test_two_grid_options_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(_MATLAB + _ELMER)
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
    """--is-partitioned-elmer-mesh requires both --elmer-mesh and --netcdf-mesh."""

    def test_partitioned_without_elmer_mesh_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(_MATLAB + ["--is-partitioned-elmer-mesh"])
        self.assertEqual(ctx.exception.code, 2)

    def test_partitioned_without_netcdf_mesh_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_cli_args(["--elmer-mesh", "mesh/", "--elmer-mesh-crs-epsg", "3413", "--is-partitioned-elmer-mesh"])
        self.assertEqual(ctx.exception.code, 2)

    def test_partitioned_with_elmer_and_netcdf_accepted(self):
        args = parse_cli_args(
            [
                "--elmer-mesh",
                "mesh/",
                "--elmer-mesh-crs-epsg",
                "3413",
                "--netcdf-mesh",
                "mesh.nc",
                "--is-partitioned-elmer-mesh",
            ]
        )
        self.assertTrue(args.is_partitioned_elmer_mesh)
        self.assertEqual(args.netcdf_mesh, Path("mesh.nc"))


class TestDefaults(unittest.TestCase):
    """Verify default values match CliDefaults."""

    def setUp(self):
        self.args = parse_cli_args(_MATLAB)

    def test_start_time_default(self):
        expected = CliDefaults.START_TIME.value.strftime("%d-%b-%Y %H:%M")
        self.assertEqual(self.args.start_time, expected)

    def test_end_time_default(self):
        expected = CliDefaults.END_TIME.value.strftime("%d-%b-%Y %H:%M")
        self.assertEqual(self.args.end_time, expected)

    def test_time_step_default(self):
        self.assertEqual(self.args.time_step, CliDefaults.TIME_STEP_SIZE_IN_DAYS.value)

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

    def test_no_coupling_by_default(self):
        self.assertFalse(self.args.couple_to_elmer_ice)
        self.assertFalse(self.args.couple_to_icon_atmo)

    def test_netcdf_mesh_default_none(self):
        self.assertIsNone(self.args.netcdf_mesh)

    def test_random_seed_default_none(self):
        self.assertIsNone(self.args.random_seed)


class TestLocalGroupLabel(unittest.TestCase):
    """--local-group-label overrides the default derived from --component-name."""

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

    def test_coupler_config(self):
        args = _parse("--coupler-config", "coupler.yaml")
        self.assertIn("--coupler-config", extract_active_coupling_features(args))

    def test_multiple_coupling_flags(self):
        args = _parse("--couple-to-elmer-ice", "--couple-to-icon-atmo")
        features = extract_active_coupling_features(args)
        self.assertIn("--couple-to-elmer-ice", features)
        self.assertIn("--couple-to-icon-atmo", features)


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
        self.assertIn("--calendar", help_text)


class TestTimeConfigCalendar(unittest.TestCase):
    def test_time_config_uses_calendar_from_cli_args(self):
        args = parse_cli_args(_MATLAB + ["--calendar", Calendar.YEAR_OF_365_DAYS.value])
        time_config = TimeConfig(args)
        self.assertEqual(time_config.calendar, Calendar.YEAR_OF_365_DAYS)


if __name__ == "__main__":
    unittest.main()
