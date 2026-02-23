# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
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


class TestPEP604UnionSyntax(unittest.TestCase):
    """Test PEP 604 union syntax (| operator) compatibility."""

    def test_pipe_union_syntax(self):
        """Test if code uses PEP 604 syntax (| for unions).

        This will fail on Python < 3.10 if the new syntax is used.
        """
        if sys.version_info < (3, 10):
            # On Python 3.9, trying to use | with types should fail
            with self.assertRaises(TypeError):
                # This would fail if evaluated as type annotation
                result = str | int  # noqa: F841
        else:
            # On Python 3.10+, this should work
            result = str | int
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
