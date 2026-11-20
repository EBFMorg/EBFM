# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Content partially generated with the assistance of AI tools.
# Claude Code: Opus 5.5

"""The sub-step limit of heat conduction in LOOP_SNOW.

A layer so thin that the solve would need more than MAX_HEAT_SUBSTEPS sub-steps
must stop the run with a message naming the column and the layer, instead of
hanging it.
"""

import unittest
import warnings

from ebfm.core import INIT, LOOP_SNOW, compute_backend
from ebfm.core.config import ColumnDiscretizationConfig

from test_loop_snow_layer_shift import _make_case

BAD_COLUMN, BAD_LAYER = 1, 10


def step(C, grid, OUT, IN):
    phys = {"snow_compaction": "firn+snow", "percolation": "normal"}
    LOOP_SNOW.main(C, OUT, IN, 1.0 / 24.0, grid, phys, ColumnDiscretizationConfig())


class TestHeatConductionSubsteps(unittest.TestCase):
    def setUp(self):
        self.C = INIT.init_constants()
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def tearDown(self):
        compute_backend._backend = compute_backend.ComputeBackend.NUMPY

    def _backends(self):
        backends = [compute_backend.ComputeBackend.NUMPY]
        if compute_backend._NUMBA_AVAILABLE:
            backends.append(compute_backend.ComputeBackend.NUMBA)
        return backends

    def test_near_zero_thickness_is_reported(self):
        for backend in self._backends():
            with self.subTest(backend=backend.name):
                compute_backend._backend = backend
                grid, OUT, IN = _make_case(self.C)
                OUT["subZ"][BAD_COLUMN, BAD_LAYER] = 1e-9
                with self.assertRaises(AssertionError) as raised:
                    step(self.C, grid, OUT, IN)
                message = str(raised.exception)
                self.assertIn(f"more than {LOOP_SNOW.MAX_HEAT_SUBSTEPS} sub-steps in 1 of 2 columns", message)
                self.assertIn(f"column {BAD_COLUMN}: dt_stab=", message)
                self.assertIn(f"thinnest layer {BAD_LAYER}: subZ=", message)

    def test_healthy_column_passes(self):
        for backend in self._backends():
            with self.subTest(backend=backend.name):
                compute_backend._backend = backend
                grid, OUT, IN = _make_case(self.C)
                step(self.C, grid, OUT, IN)


if __name__ == "__main__":
    unittest.main()
