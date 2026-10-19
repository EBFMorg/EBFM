# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

"""Tests for resolve_numba_threads().

Numba caps numba.set_num_threads() at numba.config.NUMBA_NUM_THREADS, which it derives from the CPU
affinity of the process or the NUMBA_NUM_THREADS environment variable. The CPU affinity, the MPI
layout and Numba's limit are patched, so the tests do not depend on the machine running them.
"""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

from ebfm.core import compute_backend
from ebfm.core.cli import parse_cli_args, resolve_numba_threads

try:
    import numba

    _NUMBA = True
except ImportError:  # pragma: no cover, numba is an optional dependency
    _NUMBA = False

_MATLAB = ["--matlab-mesh", "mesh.mat", "--with-numba"]


class _FakeComm:
    def __init__(self, size: int):
        self.rank = 0
        self.size = size


@unittest.skipUnless(_NUMBA, "requires numba (pip install 'ebfm[performance]')")
class TestResolveNumbaThreads(unittest.TestCase):
    def _resolve(
        self,
        *extra: str,
        numba_limit: int = 8,
        affinity: frozenset[int] | None = frozenset(range(8)),
        node_affinities: list[frozenset[int] | None] | None = None,
        comm_size: int = 1,
        env: dict[str, str] | None = None,
    ):
        """Run resolve_numba_threads() with patched CPU affinity, MPI layout and Numba limit.

        @returns (n_threads, logger)
        """
        args = parse_cli_args(_MATLAB + list(extra))
        logger = MagicMock()
        if node_affinities is None:
            node_affinities = [affinity]

        environ = {k: v for k, v in os.environ.items() if k != "NUMBA_NUM_THREADS"}
        environ.update(env or {})
        patches = [
            patch.object(numba.config, "NUMBA_NUM_THREADS", numba_limit),
            patch.object(compute_backend, "_affinity_cpus", return_value=affinity),
            patch.object(compute_backend, "_node_local_affinities", return_value=node_affinities),
            patch.dict(os.environ, environ, clear=True),
        ]

        for p in patches:
            p.start()
        try:
            n_threads = resolve_numba_threads(args, _FakeComm(comm_size), logger)
        finally:
            for p in reversed(patches):
                p.stop()
        return n_threads, logger

    def _resolve_error(self, *extra: str, **kwargs) -> str:
        """Run _resolve() expecting a CLI error, and return the error message."""
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self._resolve(*extra, **kwargs)
        return stderr.getvalue()

    def test_default_is_one_thread(self):
        n_threads, logger = self._resolve()
        self.assertEqual(n_threads, 1)
        logger.warning.assert_not_called()
        self.assertIn("suggested max 8", logger.info.call_args.args[0])

    def test_valid_thread_count(self):
        n_threads, logger = self._resolve("--numba-threads", "4")
        self.assertEqual(n_threads, 4)
        logger.warning.assert_not_called()

    def test_zero_threads_rejected(self):
        self.assertIn("must be >= 1", self._resolve_error("--numba-threads", "0"))

    def test_above_numba_limit_from_affinity_is_error(self):
        # e.g. OpenMPI binding a rank to one hyperthreaded core
        msg = self._resolve_error("--numba-threads", "8", numba_limit=2, affinity={0, 1})
        self.assertIn("exceeds Numba's thread limit of 2", msg)
        self.assertIn("may only run on 2 CPUs", msg)
        self.assertIn("--bind-to none", msg)
        self.assertIn("--cpus-per-task", msg)
        self.assertNotIn("NUMBA_NUM_THREADS=", msg)

    def test_above_numba_limit_from_env_var_names_it(self):
        msg = self._resolve_error("--numba-threads", "4", numba_limit=2, env={"NUMBA_NUM_THREADS": "2"})
        self.assertIn("exceeds Numba's thread limit of 2", msg)
        self.assertIn("NUMBA_NUM_THREADS=2", msg)

    def test_above_affinity_within_numba_limit_warns(self):
        # NUMBA_NUM_THREADS raised above the CPUs of this process: Numba allows it, EBFM warns.
        n_threads, logger = self._resolve(
            "--numba-threads", "8", numba_limit=16, affinity={0, 1, 2, 3}, env={"NUMBA_NUM_THREADS": "16"}
        )
        self.assertEqual(n_threads, 8)
        self.assertIn("exceeds the suggested max 4", logger.warning.call_args.args[0])

    def test_mpi_unbound_ranks_share_cpus(self):
        # Two ranks on this node, both allowed on all 8 CPUs: suggest 4 each, warn above that.
        all_cpus = set(range(8))
        n_threads, logger = self._resolve(
            "--numba-threads", "6", affinity=all_cpus, node_affinities=[all_cpus, all_cpus], comm_size=2
        )
        self.assertEqual(n_threads, 6)
        self.assertIn("exceeds the suggested max 4", logger.warning.call_args.args[0])
        self.assertIn("ranks on this node 2", logger.info.call_args.args[0])

    def test_mpi_bound_ranks_keep_their_cpus(self):
        # Two ranks bound to disjoint halves: each rank's own CPUs are its share, no further division.
        n_threads, logger = self._resolve(
            "--numba-threads",
            "4",
            numba_limit=4,
            affinity={0, 1, 2, 3},
            node_affinities=[{0, 1, 2, 3}, {4, 5, 6, 7}],
            comm_size=2,
        )
        self.assertEqual(n_threads, 4)
        logger.warning.assert_not_called()


class TestUsableCpuCount(unittest.TestCase):
    def test_uses_affinity(self):
        with (
            patch.object(compute_backend, "_affinity_cpus", return_value={0, 3}),
            patch("os.cpu_count", return_value=256),
        ):
            self.assertEqual(compute_backend._usable_cpu_count(), 2)

    def test_falls_back_without_affinity(self):
        with (
            patch.object(compute_backend, "_affinity_cpus", return_value=None),
            patch.object(compute_backend.os, "process_cpu_count", return_value=6, create=True),
        ):
            self.assertEqual(compute_backend._usable_cpu_count(), 6)


if __name__ == "__main__":
    unittest.main()
