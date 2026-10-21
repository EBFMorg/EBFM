# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file was generated with the help of AI tools.

"""Tests for the console log level that setup_logging() gives each MPI rank."""

import io
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ebfm.core.logger import setup_logging


class _FakeComm:
    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.size = size


class TestSetupLoggingConsoleLevel(unittest.TestCase):
    def _console_levels(self, rank: int, size: int, **kwargs) -> tuple[set[int], set[int]]:
        """Set up logging on the given rank, log one message per level and return the levels on stdout and stderr."""
        root = logging.getLogger()
        saved_handlers, saved_level = list(root.handlers), root.level
        root.handlers.clear()
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                setup_logging(comm=_FakeComm(rank, size), **kwargs)
                for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
                    logging.getLogger("ebfm.test_logger").log(level, "message")
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.close()
            root.handlers.extend(saved_handlers)
            root.setLevel(saved_level)

        def levels(text: str) -> set[int]:
            return {logging.getLevelName(name) for name in ("DEBUG", "INFO", "WARNING", "ERROR") if f" {name} " in text}

        return levels(stdout.getvalue()), levels(stderr.getvalue())

    def test_console_level_per_rank(self):
        cases = {
            "serial run": dict(rank=0, size=1, stdout_log_level=logging.INFO, expected=logging.INFO),
            "rank 0 of a parallel run": dict(rank=0, size=4, stdout_log_level=logging.INFO, expected=logging.INFO),
            "rank 0 at DEBUG": dict(rank=0, size=4, stdout_log_level=logging.DEBUG, expected=logging.DEBUG),
            "other rank": dict(rank=2, size=4, stdout_log_level=logging.INFO, expected=logging.WARNING),
            "other rank, DEBUG raised to WARNING": dict(
                rank=2, size=4, stdout_log_level=logging.DEBUG, expected=logging.WARNING
            ),
            "other rank, ERROR kept": dict(rank=2, size=4, stdout_log_level=logging.ERROR, expected=logging.ERROR),
            "individual level of other rank": dict(
                rank=2,
                size=4,
                stdout_log_level=logging.INFO,
                stdout_individual_log_levels={2: logging.DEBUG},
                expected=logging.DEBUG,
            ),
            "individual level of rank 0": dict(
                rank=0,
                size=4,
                stdout_log_level=logging.INFO,
                stdout_individual_log_levels={0: logging.WARNING},
                expected=logging.WARNING,
            ),
            "individual level for a different rank": dict(
                rank=1,
                size=4,
                stdout_log_level=logging.INFO,
                stdout_individual_log_levels={2: logging.DEBUG},
                expected=logging.WARNING,
            ),
        }
        for name, case in cases.items():
            with self.subTest(name):
                case = dict(case)
                expected = case.pop("expected")
                case.setdefault("stdout_individual_log_levels", {})
                stdout_levels, stderr_levels = self._console_levels(**case)
                self.assertEqual(
                    stdout_levels,
                    {level for level in (logging.DEBUG, logging.INFO, logging.WARNING) if level >= expected},
                )
                self.assertEqual(stderr_levels, {logging.ERROR})


if __name__ == "__main__":
    unittest.main()
