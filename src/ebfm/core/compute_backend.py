# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Compute-backend dispatch for EBFM kernel functions.

Manages which compute backend is active.
Availability: NumPy (standard), Numba (optional, for CPU parallelism).
In the future: GPU offloading (e.g. with CuPy or Numba CUDA).

Usage in main.py:
    from ebfm.core.compute_backend import init_numba, is_numba_available

    if args.with_numba:
        init_numba(n_threads)

Usage in LOOP_SNOW.py (dispatch):
    from .compute_backend import get_backend, ComputeBackend

    if get_backend() == ComputeBackend.NUMBA:
        _compaction_kernel(...)
    else:
        # standard NumPy path
        ...
"""

from enum import Enum


class ComputeBackend(Enum):
    NUMPY = "numpy"
    NUMBA = "numba"
    # CUDA = "cuda"  # future


_backend = ComputeBackend.NUMPY

# ---------------------------------------------------------------------------
# Numba availability and decorator definitions
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange  # noqa: F401

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # noqa: E302
        """No-op replacement for numba.njit when numba is not installed."""

        def _wrap(fn):
            return fn

        return _wrap if kwargs or (args and not callable(args[0])) else args[0]

    prange = range  # type: ignore[assignment]  # noqa: F811


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_backend() -> ComputeBackend:
    """Return the currently active compute backend."""
    return _backend


def is_numba_available() -> bool:
    """Return True if numba is installed and importable."""
    return _NUMBA_AVAILABLE


def init_numba(n_threads: int = 1):
    """Activate the Numba backend with the given thread count.

    Must be called before any kernel runs (i.e. before the time loop).
    """
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not installed. Run: pip install 'ebfm[performance]'")

    import numba

    numba.set_num_threads(n_threads)
    assert numba.get_num_threads() >= 1

    global _backend
    _backend = ComputeBackend.NUMBA
