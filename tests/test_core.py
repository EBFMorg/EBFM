# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from ebfm.core import get_version


class TestCore(unittest.TestCase):
    """Test cases for core module functions."""

    def test_get_version(self):
        """Test that get_version returns a valid version string."""
        version = get_version()

        # Check that version is a string
        self.assertIsInstance(version, str)

        # Check that version is not empty
        self.assertGreater(len(version), 0)

        # Check that version is either a proper version format or "unknown"
        self.assertTrue(
            version == "unknown" or any(char.isdigit() for char in version),
            f"Version '{version}' should be 'unknown' or contain digits",
        )


class TestPEP604SyntaxStandalone(unittest.TestCase):
    """
    PEP 604 (X | Y) union type annotations require Python >= 3.10.

    This test documents the intentional drop of Python 3.9 support.
    """

    def test_pep604_type_annotation(self):
        code = """
from __future__ import annotations
from typing import get_type_hints

def foo(x: int | str) -> None:
    pass
"""

        namespace = {}
        exec(code, namespace)

        foo = namespace["foo"]

        from typing import get_type_hints

        hints = get_type_hints(foo)
        self.assertEqual(hints["x"], int | str)


if __name__ == "__main__":
    unittest.main()
