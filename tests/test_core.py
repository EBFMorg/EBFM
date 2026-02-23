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

class TestImports(unittest.TestCase):
    """Import tests to ensure modules are importable in supported Python versions."""

    def test_import_yac_coupler(self):
        """Import yacCoupler (will fail on Python < 3.10 if PEP604 syntax is used)."""
        import ebfm.coupling.couplers.yacCoupler  # noqa: F401

if __name__ == "__main__":
    unittest.main()
