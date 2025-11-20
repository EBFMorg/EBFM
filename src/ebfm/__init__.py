# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("EBFM")
except PackageNotFoundError:
    __version__ = "unknown"


def get_version() -> str:
    """Return the current version of the EBFM package as a string."""
    return __version__


def print_version_and_exit():
    """Print the current version of the EBFM package and exit."""
    print(f"{get_version()}")
    exit(0)
