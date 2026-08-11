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
    GPU = "gpu"  # numba.cuda (NVIDIA) or numba.hip (AMD)


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
# GPU (CUDA / ROCm-HIP) availability and the `cuda` handle used to decorate
# GPU kernels (@cuda.jit) in LOOP_SNOW_gpu_kernels.py.
#
# The same numba.cuda kernels also run on AMD GPUs: numba.hip.pose_as_cuda()
# makes `from numba import cuda` delegate to HIP.  Vendor detection therefore
# happens at IMPORT time, because kernel decoration is import-time.
#
# _GPU_AVAILABLE only means the GPU stack is importable. Presence of a usable
# device is confirmed separately by the smoke test in init_gpu().
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
_GPU_VENDOR: str | None = None  # "nvidia" | "amd" | None

try:
    # AMD path: numba-hip provides `numba.hip`; pose_as_cuda() makes
    # `from numba import cuda` kernels run on gfx GPUs unchanged.
    from numba import hip as _hip

    _hip.pose_as_cuda()
    from numba import cuda  # noqa: F401  (now backed by HIP)

    _GPU_AVAILABLE = True
    _GPU_VENDOR = "amd"
except ImportError:
    try:
        # NVIDIA path: numba.cuda (bundled numba-cuda for numba >= 0.62).
        from numba import cuda  # noqa: F401

        _GPU_AVAILABLE = True
        _GPU_VENDOR = "nvidia"
    except ImportError:
        _GPU_AVAILABLE = False
        _GPU_VENDOR = None

        class _CudaStub:
            """No-op stand-in so @cuda.jit stays importable without a GPU stack."""

            @staticmethod
            def jit(fn=None, **kwargs):
                if fn is not None:
                    return fn

                def _wrap(f):
                    return f

                return _wrap

            @staticmethod
            def grid(*args):
                return 0

        cuda = _CudaStub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_backend() -> ComputeBackend:
    """Return the currently active compute backend."""
    return _backend


def is_numba_available() -> bool:
    """Return True if numba is installed and importable."""
    return _NUMBA_AVAILABLE


def is_gpu_available() -> bool:
    """Return True if a GPU stack (numba.cuda or numba.hip) is importable.

    Note: this does not guarantee a usable device is present. That is verified
    by the smoke test run inside init_gpu().
    """
    return _GPU_AVAILABLE


def get_gpu_vendor() -> str | None:
    """Return the detected GPU vendor ("nvidia", "amd") or None if unavailable."""
    return _GPU_VENDOR


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


def init_gpu(vendor: str = "auto", logger=None):
    """Activate the GPU backend after confirming a usable device.

    Runs a small round-trip smoke test to verify the device works before
    switching the backend. Must be called before any kernel runs (i.e. before
    the time loop).

    @param vendor "auto" | "nvidia" | "amd"; when not "auto", guards against a
                  mismatch with the import-time detected GPU stack.
    @param logger optional logger for the device info message.
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            "GPU backend unavailable: could not import numba.cuda (NVIDIA) or "
            "numba.hip (AMD). Install via: pip install 'ebfm[gpu]' and load the "
            "CUDA/ROCm toolkit on the compute node."
        )

    if vendor != "auto" and vendor != _GPU_VENDOR:
        raise RuntimeError(
            f"--gpu-vendor={vendor} requested but the detected GPU stack is "
            f"'{_GPU_VENDOR}'. Load the matching toolkit or use --gpu-vendor auto."
        )

    # Imported lazily to avoid a circular import at module load time.
    from ebfm.core.LOOP_SNOW_gpu_kernels import gpu_offload_smoke_test

    smoke = gpu_offload_smoke_test()
    if not smoke.get("available", False):
        raise RuntimeError(f"GPU smoke test failed: {smoke.get('reason', 'unknown error')}")

    global _backend
    _backend = ComputeBackend.GPU

    if logger is not None:
        logger.info(
            "[GPU] backend enabled (%s). Device: %s  free=%.2f GiB  total=%.2f GiB",
            _GPU_VENDOR,
            smoke["device_name"],
            smoke["free_mem_gb"],
            smoke["total_mem_gb"],
        )
