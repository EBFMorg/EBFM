# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ebfm.core import LOOP_write_to_file
from ebfm.core.LOOP_write_to_file import _output_dimensions, main as write_output
from ebfm.core.config import ColumnDiscretizationConfig

NL = 4
NY, NX = 3, 4
# Flat indices of the glacier points inside the NY x NX structured grid.
GLACIER_INDICES = (0, 3, 5, 7, 11)
GPSUM = len(GLACIER_INDICES)

_NETCDF_OUTPUT = 2


def _make_column():
    return ColumnDiscretizationConfig(nl=NL, split=(2,))


class _AutoOut(dict):
    """
    An OUT dict that fabricates plausible data for whatever the writer asks for.

    The writer declares its own variable list, so deriving the data from the
    requested key keeps these tests exercising every output variable, including
    ones added later.
    """

    def __init__(self, gpsum, nl):
        super().__init__()
        self._gpsum = gpsum
        self._nl = nl

    def __missing__(self, key):
        if key.startswith("sub"):
            value = np.arange(self._gpsum * self._nl, dtype=np.float64).reshape(self._gpsum, self._nl)
        elif key == "is_shaded":
            value = np.zeros(self._gpsum, dtype=bool)
        else:
            value = np.arange(self._gpsum, dtype=np.float64)
        self[key] = value
        return value


def _structured_grid():
    x_2D, y_2D = np.meshgrid(np.arange(NX, dtype=np.float64), np.arange(NY, dtype=np.float64))
    return {
        "is_unstructured": False,
        "is_partitioned": False,
        "x_2D": x_2D,
        "y_2D": y_2D,
        "z_2D": np.full((NY, NX), 100.0),
        "lon_2D": np.zeros((NY, NX)),
        "lat_2D": np.zeros((NY, NX)),
        "ind": np.array(GLACIER_INDICES),
    }


def _unstructured_grid():
    return {
        "is_unstructured": True,
        "is_partitioned": False,
        "x": np.arange(GPSUM, dtype=np.float64),
        "y": np.arange(GPSUM, dtype=np.float64),
        "z": np.full(GPSUM, 100.0),
        "lon": np.zeros(GPSUM),
        "lat": np.zeros(GPSUM),
    }


def _write_single_step(outdir, grid):
    """
    Run the writer for a one-step simulation and return the closed output file.

    With freqout=1 and tn=1 the single call at t=0 creates the file, defines the
    variables, writes time index 0 and closes the file again.
    """
    column = _make_column()
    out = _AutoOut(GPSUM, NL)
    io = {"outdir": str(outdir), "freqout": 1, "output_type": _NETCDF_OUTPUT}
    time = {"TCUR": datetime(2020, 1, 1), "tn": 1}

    io, outfile = write_output({}, io, out, grid, 0, time, column)

    return Path(outdir) / "model_output.nc", outfile, out


class TestNetCDFOutputSmoke(unittest.TestCase):
    """
    End-to-end checks for the NetCDF output path (io["output_type"] == 2).

    This is the default output type set in INIT, and the writer takes the column
    discretization as a separate argument from the grid. These tests drive the
    real entry point so that a helper whose signature or arguments drift out of
    sync fails here instead of at runtime.
    """

    def test_structured_grid_writes_all_declared_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, outfile, _ = _write_single_step(tmp, _structured_grid())

            self.assertTrue(path.is_file())
            with Dataset(path) as nc:
                self.assertEqual(len(nc.dimensions["nl"]), NL)
                self.assertEqual(len(nc.dimensions["y"]), NY)
                self.assertEqual(len(nc.dimensions["x"]), NX)

                for entry in outfile["varsout"]:
                    varname = entry[0]
                    self.assertIn(varname, nc.variables)
                    if varname.startswith("sub"):
                        self.assertEqual(nc[varname].dimensions, ("time", "y", "x", "nl"))
                        self.assertEqual(nc[varname].shape, (1, NY, NX, NL))
                    else:
                        self.assertEqual(nc[varname].dimensions, ("time", "y", "x"))
                        self.assertEqual(nc[varname].shape, (1, NY, NX))

    def test_unstructured_grid_writes_all_declared_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, outfile, _ = _write_single_step(tmp, _unstructured_grid())

            self.assertTrue(path.is_file())
            with Dataset(path) as nc:
                self.assertEqual(len(nc.dimensions["nl"]), NL)
                self.assertEqual(len(nc.dimensions["cell"]), GPSUM)

                for entry in outfile["varsout"]:
                    varname = entry[0]
                    self.assertIn(varname, nc.variables)
                    if varname.startswith("sub"):
                        self.assertEqual(nc[varname].dimensions, ("time", "cell", "nl"))
                        self.assertEqual(nc[varname].shape, (1, GPSUM, NL))
                    else:
                        self.assertEqual(nc[varname].dimensions, ("time", "cell"))
                        self.assertEqual(nc[varname].shape, (1, GPSUM))

    def test_structured_grid_scatters_values_onto_glacier_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, out = _write_single_step(tmp, _structured_grid())

            with Dataset(path) as nc:
                # A 3D (time, y, x) variable: glacier points carry data, the rest is masked.
                smb = nc["smb"][0]
                np.testing.assert_allclose(smb.compressed(), out["smb"])
                self.assertEqual(smb.count(), GPSUM)

                # A 4D (time, y, x, nl) variable: every layer of every glacier column.
                sub_t = nc["subT"][0]
                flat = sub_t.reshape(NY * NX, NL)
                np.testing.assert_allclose(flat[list(GLACIER_INDICES), :], out["subT"])
                self.assertEqual(sub_t.count(), GPSUM * NL)

    def test_layer_count_comes_from_the_column_not_the_grid(self):
        """
        The number of layers lives in the column discretization; the grid dict
        carries no "nl" key. Passing a grid without one must still work.
        """
        column = _make_column()
        grid = _structured_grid()
        self.assertNotIn("nl", grid)

        dimensions, chunksizes = _output_dimensions("subT", grid, column)
        self.assertEqual(dimensions, ("time", "y", "x", "nl"))
        self.assertEqual(chunksizes, (1, NY, NX, column.nl))

        unstructured = _unstructured_grid()
        self.assertNotIn("nl", unstructured)

        dimensions, chunksizes = _output_dimensions("subT", unstructured, column)
        self.assertEqual(dimensions, ("time", "cell", "nl"))
        self.assertEqual(chunksizes, (1, GPSUM, column.nl))

    def test_time_variable_records_the_current_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = _write_single_step(tmp, _structured_grid())

            with Dataset(path) as nc:
                self.assertEqual(nc["time"].units, "days since 1970-01-01 00:00:00")
                # 2020-01-01 is 18262 days after the epoch.
                self.assertAlmostEqual(float(nc["time"][0]), 18262.0)

    def test_every_module_helper_taking_a_column_is_called_with_one(self):
        """
        Guard against the writer's helpers drifting out of sync again: any helper
        that declares a `column` parameter must receive it at every call site.
        """
        import ast
        import inspect

        source = inspect.getsource(LOOP_write_to_file)
        tree = ast.parse(source)

        column_helpers = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "column" in {a.arg for a in node.args.args}
        }
        self.assertIn("_output_dimensions", column_helpers)
        self.assertIn("_write_output_variable", column_helpers)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            helper = column_helpers.get(node.func.id)
            if helper is None:
                continue
            passed = len(node.args) + len(node.keywords)
            self.assertEqual(
                passed,
                len(helper.args.args),
                f"{node.func.id}() is called with {passed} arguments on line {node.lineno} "
                f"but declares {len(helper.args.args)}",
            )


if __name__ == "__main__":
    unittest.main()
