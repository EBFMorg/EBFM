# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GPU kernels for the LOOP_SNOW module.

Availability: numba.cuda kernels for all six compute sub-steps of LOOP_SNOW --
snowfall/deposition, melt/sublimation, compaction, heat conduction,
percolation and layer merging/splitting -- offloaded to a GPU when the GPU
backend is active (--with-gpu). Only runoff(), which is (gpsum,) vector
arithmetic on values already downloaded, stays on the host.

The same kernels run on both vendors:
- NVIDIA via stock numba.cuda,
- AMD via numba.hip.pose_as_cuda() (see compute_backend.py).

Each ``_*_kernel_gpu`` is the CUDA-thread equivalent of the corresponding
``@njit`` kernel in LOOP_SNOW_kernels.py: the outer ``prange(gpsum)`` loop is
replaced by a single CUDA thread index so one GPU thread handles one grid
column. Per-column scratch arrays (suffix ``_ws``, shape ``(gpsum, nl)``) are
pre-allocated on the device by ``SnowDeviceState`` to avoid dynamic allocation
inside the kernels.

``SnowDeviceState`` owns the device buffers: it uploads the subsurface state
once per run, launches every sub-step kernel against the resident arrays, and
copies back only what host code reads between timesteps. The host-side physics
glue lives in LOOP_SNOW.py (shared with the NumPy / Numba paths), so results
match those paths exactly.

When no GPU stack is importable, ``cuda`` is a no-op stub (from
compute_backend), so the @cuda.jit-decorated functions remain importable but
are never launched.
"""

import math
import os

import numpy as np

from .compute_backend import cuda

# Threads per block used for all 1-D column-parallel launches.
#
# The best value is device- and kernel-dependent (these kernels are large, so
# register pressure can limit occupancy). Override for experiments with
# EBFM_GPU_TPB, e.g. EBFM_GPU_TPB=256.
_TPB = int(os.environ.get("EBFM_GPU_TPB", "128"))


def _blocks(gpsum: int) -> int:
    """Number of blocks needed to cover ``gpsum`` columns at ``_TPB`` threads each."""
    return (gpsum + _TPB - 1) // _TPB


# ---------------------------------------------------------------------------
# Device helper functions
#
# Python's built-in max(), min(), abs() cannot be reliably used inside
# @cuda.jit kernels across all numba/Python versions.  In Python 3.12+
# (and especially 3.14) numba's CUDA type-inference maps them to a
# single-argument overload, raising "2 argument types given, but function
# takes 1 arguments" at JIT-compile time.  The CPU @njit path handles them
# via a separate overload mechanism and is unaffected.
#
# @cuda.jit(device=True) functions are the canonical numba solution: they
# are compiled once for the device, inlined at every call site, and work
# on all supported numba versions and Python versions.
#
# When the GPU stack is absent (_CudaStub from compute_backend), the
# decorator is a no-op, so the functions stay importable as plain Python.
# ---------------------------------------------------------------------------


@cuda.jit(device=True)
def _fmax(a, b):
    """max(a, b) for use inside GPU kernels (replaces Python built-in)."""
    return a if a > b else b


@cuda.jit(device=True)
def _fmin(a, b):
    """min(a, b) for use inside GPU kernels (replaces Python built-in)."""
    return a if a < b else b


@cuda.jit(device=True)
def _fabs(x):
    """abs(x) for use inside GPU kernels (replaces Python built-in)."""
    return x if x >= 0.0 else -x


# ===========================================================================
# GPU kernels
# ===========================================================================


@cuda.jit
def _gpu_probe_kernel(arr):
    """Trivial kernel used only by gpu_offload_smoke_test().

    Multiplies every element of ``arr`` by 2 to verify a GPU round-trip.
    """
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] *= 2.0


@cuda.jit
def _snowfall_prep_kernel_gpu(
    subZ,
    Dfreshsnow,
    Dfreshsnow_T,
    Dfreshsnow_W,
    shift_tot,
    surfH,
    runoff_irr_deep,
    T,  # (gpsum,) air temperature
    WS,  # (gpsum,) wind speed
    snow,  # (gpsum,) snowfall
    moist_deposition,  # (gpsum,) riming
    Dwater,
    T0,
    Dfreshsnow_const,
    compaction_mode,
    more_flag,
):
    """Fresh-snow density + total grid shift. One CUDA thread per grid column.

    Device equivalent of the block preceding the ``while np.any(shift_tot > 0)``
    loop in snowfall_and_deposition(). ``more_flag`` receives the device-side
    ``np.any(shift_tot > 0)`` that drives the host loop.
    """
    i = cuda.grid(1)
    if i >= subZ.shape[1]:
        return

    if compaction_mode == 1:  # firn+snow: temperature/wind dependent fresh-snow density
        t_i = T[i]
        if t_i > T0 + 2.0:
            d_t = 50.0 + 1.7 * 17.0**1.5
        elif t_i > T0 - 15.0:
            d_t = 50.0 + 1.7 * (t_i - T0 + 15.0) ** 1.5
        else:
            d_t = -3.8328 * (t_i - T0) - 0.0333 * (t_i - T0) ** 2
        d_w = 266.86 * (0.5 * (1.0 + math.tanh(WS[i] / 5.0))) ** 8.8
        Dfreshsnow_T[i] = d_t
        Dfreshsnow_W[i] = d_w
        Dfreshsnow[i] = d_t + d_w
    else:
        # The NumPy path leaves the two components undefined in firn_only mode;
        # zero them rather than let download_state() hand back uninitialised
        # device memory.
        Dfreshsnow_T[i] = 0.0
        Dfreshsnow_W[i] = 0.0
        Dfreshsnow[i] = Dfreshsnow_const

    d_fresh = Dfreshsnow[i]
    shift_i = snow[i] * Dwater / d_fresh + moist_deposition[i] * Dwater / d_fresh
    shift_tot[i] = shift_i
    surfH[i] += shift_i
    runoff_irr_deep[i] = 0.0

    if shift_i > 0.0:
        cuda.atomic.max(more_flag, 0, 1)


@cuda.jit
def _snowfall_shift_kernel_gpu(
    subZ,
    subT,
    subD,
    subW,
    Dfreshsnow,
    Tsurf,
    shift_tot,
    runoff_irr_deep,
    max_subZ,
    more_flag,
):
    """One pass of the snowfall grid-shift loop. One CUDA thread per column.

    Device equivalent of the body of ``while np.any(shift_tot > 0)`` in
    snowfall_and_deposition(). The host re-launches this kernel while
    ``more_flag`` stays set, so every column executes exactly as many passes as
    on the NumPy path (columns that are already finished see ``shift == 0`` and
    run the same no-op arithmetic, which keeps the results bit-identical).
    """
    i = cuda.grid(1)
    if i >= subZ.shape[1]:
        return
    nl = subZ.shape[0]

    shift = _fmin(shift_tot[i], max_subZ)
    shift_tot[i] -= shift

    z0_old = subZ[0, i]
    t0_old = subT[0, i]
    d0_old = subD[0, i]

    if z0_old + shift <= max_subZ:
        # ------ no-shift branch: the new snow is absorbed by the top layer ------
        z0_new = z0_old + shift
        subZ[0, i] = z0_new
        subT[0, i] = t0_old * z0_old / z0_new + Tsurf[i] * shift / z0_new
        subD[0, i] = d0_old * z0_old / z0_new + Dfreshsnow[i] * shift / z0_new
    else:
        # ------ shift branch: push the whole column down by one layer ------
        # The deepest layer is never written here, so its pre-shift water is
        # still available for the deep-runoff bookkeeping below.
        runoff_irr_deep[i] += subW[nl - 1, i]
        w0_old = subW[0, i]

        # subZ[2:nl-1] = subZ_old[1:nl-2]; writing high-to-low keeps the source
        # values intact, so no column snapshot is needed.
        for k in range(nl - 2, 1, -1):
            subZ[k, i] = subZ[k - 1, i]
            subT[k, i] = subT[k - 1, i]
            subD[k, i] = subD[k - 1, i]
            subW[k, i] = subW[k - 1, i]

        z1_new = max_subZ
        subZ[1, i] = z1_new
        subZ[0, i] = (z0_old + shift) - max_subZ
        subT[1, i] = t0_old * z0_old / z1_new + Tsurf[i] * (z1_new - z0_old) / z1_new
        subT[0, i] = Tsurf[i]
        subD[1, i] = d0_old * z0_old / z1_new + Dfreshsnow[i] * (z1_new - z0_old) / z1_new
        subD[0, i] = Dfreshsnow[i]
        subW[1, i] = w0_old
        subW[0, i] = 0.0

    if shift_tot[i] > 0.0:
        cuda.atomic.max(more_flag, 0, 1)


@cuda.jit
def _melt_peel_kernel_gpu(subZ, subD, subW, melt, moist_sublimation, sumWinit, shift_tot, more_flag):
    """Melt/sublimation mass removal. One CUDA thread per grid column.

    Device equivalent of ``OUT["sumWinit"] = ...`` plus the
    ``while np.any(mass_removed > 0)`` layer-peeling loop in melt_sublimation().
    That loop is a no-op for columns whose mass is already used up (both of its
    masks are False there), so it can be run per thread with a private trip
    count instead of a host-driven one. ``more_flag`` receives the device-side
    ``np.any(shift_tot < 0)`` for the shift loop that follows.
    """
    i = cuda.grid(1)
    if i >= subZ.shape[1]:
        return
    nl = subZ.shape[0]

    w_sum = 0.0
    for k in range(nl):
        w_sum += subW[k, i]
    sumWinit[i] = w_sum

    mass_removed = (melt[i] + moist_sublimation[i]) * 1e3
    shift_i = 0.0
    n = 0
    # Bounded by nl: the NumPy path would raise IndexError past the column
    # bottom, so stopping there cannot change any result it can produce.
    while mass_removed > 0.0 and n < nl:
        mass_layer = subD[n, i] * subZ[n, i]
        if mass_removed > mass_layer:
            # Layer fully removed
            mass_removed -= subD[n, i] * subZ[n, i]
            shift_i -= subZ[n, i]
        else:
            # Layer partially removed
            shift_i -= (mass_removed / mass_layer) * subZ[n, i]
            mass_removed = 0.0
        n += 1

    shift_tot[i] = shift_i
    if shift_i < 0.0:
        cuda.atomic.max(more_flag, 0, 1)


@cuda.jit
def _melt_shift_kernel_gpu(
    subZ,
    subT,
    subD,
    subW,
    surfH,
    shift_tot,
    bottom_thickness,
    more_flag,
):
    """One pass of the melt grid-shift loop. One CUDA thread per column.

    Device equivalent of the body of ``while np.any(shift_tot < 0)`` in
    melt_sublimation(); driven by the host the same way as
    _snowfall_shift_kernel_gpu.
    """
    i = cuda.grid(1)
    if i >= subZ.shape[1]:
        return
    nl = subZ.shape[0]

    shift = _fmax(shift_tot[i], -subZ[1, i])
    shift_tot[i] -= shift
    surfH[i] += shift

    z0_old = subZ[0, i]

    if z0_old + shift > 1e-17:
        # ------ no-shift branch: the top layer only gets thinner ------
        z0_new = z0_old + shift
        subZ[0, i] = z0_new
        subW[0, i] = subW[0, i] * (z0_new / z0_old)
    else:
        # ------ shift branch: layers 0 and 1 collapse into one ------
        z1_old = subZ[1, i]
        t1_old = subT[1, i]
        d1_old = subD[1, i]
        w1_old = subW[1, i]

        # subZ[1:nl-2] = subZ_old[2:nl-1]; writing low-to-high keeps the source
        # values intact, so no column snapshot is needed.
        for k in range(1, nl - 2):
            subZ[k, i] = subZ[k + 1, i]
            subT[k, i] = subT[k + 1, i]
            subD[k, i] = subD[k + 1, i]
            subW[k, i] = subW[k + 1, i]

        z0_new = z0_old + z1_old + shift
        subZ[0, i] = z0_new
        subT[0, i] = t1_old
        subD[0, i] = d1_old
        subW[0, i] = w1_old * (z0_new / z1_old)

        # The NumPy path re-assigns subT/subD of the deepest layer from its own
        # (unshifted) value, i.e. a no-op; only the thickness and water reset.
        subZ[nl - 1, i] = bottom_thickness
        subW[nl - 1, i] = 0.0

    if shift_tot[i] < 0.0:
        cuda.atomic.max(more_flag, 0, 1)


@cuda.jit
def _layer_merging_splitting_kernel_gpu(
    subZ,
    subT,
    subD,
    subW,
    subS,
    runoff_irr_deep,
    runoff_slush,
    mask,  # (gpsum,) int — 1 marks an active (glacier) column
    split_arr,  # (nsplit,) int — layer indices at which the thickness doubles
    max_subZ,
    top_thickness,  # (2**nsplit) * max_subZ
):
    """GPU equivalent of layer_merging_and_splitting(). One thread per column.

    Ported from the current CPU code in LOOP_SNOW.py, not from the AMD
    reference implementation: that one uses a linear extrapolation for the
    deepest-layer temperature after a merge, whereas the CPU path carries the
    previous deepest-layer temperature over unchanged, and it makes merge and
    split mutually exclusive, whereas the CPU path evaluates the split
    condition on the post-merge column so both can fire in the same pass.

    Both branches shift the column in the direction that lets the source values
    be read before they are overwritten (merge shifts up, so it runs
    low-to-high; split shifts down, so it runs high-to-low), which removes the
    five per-column snapshot arrays the host path allocates.
    """
    i = cuda.grid(1)
    if i >= subZ.shape[1]:
        return
    if mask[i] != 1:
        return
    nl = subZ.shape[0]

    # The host computes (2.0**n) * max_subZ; doubling gives bit-identical
    # values (scaling a double by a power of two is exact) without a pow().
    threshold = max_subZ
    for n in range(split_arr.shape[0]):
        split = split_arr[n]

        # ------ Merge layers (accumulation case) ------
        if subZ[split, i] <= threshold:
            # The new base layer keeps the old deepest layer's T/D; read them
            # before the shift below moves the column.
            t_base = subT[nl - 1, i]
            d_base = subD[nl - 1, i]
            zm = subZ[split - 1, i]
            zs = subZ[split, i]
            den = zm + zs
            subZ[split - 1, i] = den
            subW[split - 1, i] = subW[split - 1, i] + subW[split, i]
            subS[split - 1, i] = subS[split - 1, i] + subS[split, i]
            subD[split - 1, i] = (zm * subD[split - 1, i] + zs * subD[split, i]) / den
            subT[split - 1, i] = (zm * subT[split - 1, i] + zs * subT[split, i]) / den

            # Shift the layers below the merge up by one
            for k in range(split, nl - 1):
                subZ[k, i] = subZ[k + 1, i]
                subW[k, i] = subW[k + 1, i]
                subS[k, i] = subS[k + 1, i]
                subD[k, i] = subD[k + 1, i]
                subT[k, i] = subT[k + 1, i]

            # New layer at the base, inheriting the old deepest layer's T/D
            subZ[nl - 1, i] = top_thickness
            subT[nl - 1, i] = t_base
            subD[nl - 1, i] = d_base
            subW[nl - 1, i] = 0.0
            subS[nl - 1, i] = 0.0

        # ------ Split layers (ablation case), evaluated post-merge ------
        if subZ[split - 2, i] > threshold:
            # The layer pushed out of the bottom releases its water
            runoff_irr_deep[i] += subW[nl - 1, i]
            runoff_slush[i] += subS[nl - 1, i]

            # Shift down by one first, so split-1 still holds its pre-split
            # value when it is read as the shift source.
            for k in range(nl - 1, split - 1, -1):
                subZ[k, i] = subZ[k - 1, i]
                subW[k, i] = subW[k - 1, i]
                subS[k, i] = subS[k - 1, i]
                subD[k, i] = subD[k - 1, i]
                subT[k, i] = subT[k - 1, i]

            # Halve the split layer and copy it into the freed slot below
            subZ[split - 2, i] *= 0.5
            subW[split - 2, i] *= 0.5
            subS[split - 2, i] *= 0.5
            subZ[split - 1, i] = subZ[split - 2, i]
            subW[split - 1, i] = subW[split - 2, i]
            subS[split - 1, i] = subS[split - 2, i]
            subT[split - 1, i] = subT[split - 2, i]
            subD[split - 1, i] = subD[split - 2, i]

        threshold *= 2.0


@cuda.jit
def _all_ice_kernel_gpu(subD, Dice, all_ice):
    """Column-wise ``np.all(subD >= Dice, axis=1)`` as a (gpsum,) uint8 flag.

    LOOP_mass_balance is the only host code that reduces over the full density
    grid between timesteps; computing the reduction here keeps subD resident.
    """
    i = cuda.grid(1)
    if i >= subD.shape[1]:
        return
    for k in range(subD.shape[0]):
        if subD[k, i] < Dice:
            all_ice[i] = 0
            return
    all_ice[i] = 1


@cuda.jit
def _compaction_kernel_gpu(
    subD,
    subZ,
    subT,
    subW,
    subTmean,
    subD_old,
    logyearsnow,  # (gpsum,) - layer-invariant on the host, so kept 1-D here
    yearsnow,  # (gpsum,)
    WS,
    Dens_destr_metam,
    Dens_overb_pres,
    Dens_drift,
    surfH,
    sumWinit,
    runoff_irr,
    dt_yearfrac,
    dt_seconds,
    dt,
    Dice,
    Dfirn,
    Dwater,
    g,
    T0,
    rd,
    Ec,
    Eg,
    dayseconds,
    tau_drift,
    compaction_mode,
    was_snow_ws,  # (gpsum, nl) bool workspace
    psload_ws,  # (gpsum, nl) float64 workspace
):
    """GPU equivalent of _compaction_kernel. One CUDA thread per grid column."""
    i = cuda.grid(1)
    if i >= subD.shape[1]:
        return
    nl = subD.shape[0]

    # ------ 1. FIRN COMPACTION ------ #
    for k in range(nl):
        subTmean[k, i] = subTmean[k, i] * (1.0 - dt_yearfrac) + dt_yearfrac * subT[k, i]
        cond_firn_k = (compaction_mode == 0) or (subD[k, i] >= Dfirn)
        if cond_firn_k:
            if subD[k, i] < 550.0:
                grav_const = 0.07 * _fmax(1.435 - 0.151 * logyearsnow[i], 0.25)
            else:
                grav_const = 0.03 * _fmax(2.366 - 0.293 * logyearsnow[i], 0.25)
            temp_factor = math.exp(-Ec / (rd * subT[k, i]) + Eg / (rd * subTmean[k, i]))
            firn_inc = dt_yearfrac * grav_const * yearsnow[i] * g * (Dice - subD[k, i]) * temp_factor
            subD[k, i] += firn_inc

    # ------ 2. SEASONAL SNOW COMPACTION ------ #
    if compaction_mode == 1:  # firn+snow
        # Capture pre-DM snow mask
        for k in range(nl):
            was_snow_ws[k, i] = subD[k, i] < Dfirn

        # ------ 2.1 DESTRUCTIVE METAMORPHISM ------ #
        for k in range(nl):
            if was_snow_ws[k, i]:
                cc1 = math.exp(-0.046 * _fmax(subD[k, i] - 175.0, 0.0))
                cc2 = 1.0 + (1.0 if subW[k, i] != 0.0 else 0.0)
                temp_exp = math.exp(0.04 * (subT[k, i] - T0))
                snow_inc = cc1 * cc2 * 2.777e-6 * temp_exp * dt_seconds * subD[k, i]
                subD[k, i] = _fmin(subD[k, i] + snow_inc, Dice)
                Dens_destr_metam[k, i] = snow_inc
            else:
                Dens_destr_metam[k, i] = 0.0

        # ------ 2.2 OVERBURDEN PRESSURE ------ #
        psload_ws[0, i] = 0.5 * subD[0, i] * subZ[0, i] * g
        for k in range(1, nl):
            xm = subD[k - 1, i] * subZ[k - 1, i] * g
            xk = subD[k, i] * subZ[k, i] * g
            psload_ws[k, i] = psload_ws[k - 1, i] + 0.5 * (xm + xk)

        for k in range(nl):
            Dens_overb_pres[k, i] = 0.0
            if was_snow_ws[k, i]:
                cc7 = 4.0 * 7.62237e6 / 250.0 * subD[k, i] / (1.0 + 60.0 * subW[k, i] / (Dwater * subZ[k, i]))
                visc = cc7 * math.exp(0.1 * (T0 - subT[k, i]) + 0.023 * subD[k, i])
                overb_inc = dt * dayseconds * subD[k, i] * psload_ws[k, i] / visc
                subD[k, i] = _fmin(subD[k, i] + overb_inc, Dice)
                Dens_overb_pres[k, i] = dt * dayseconds * subD[k, i] * psload_ws[k, i] / visc

        # ------ 2.3 DRIFTING SNOW ------ #
        z_i_k = 0.0
        for k in range(nl):
            d_k = _fmax(subD[k, i], 50.0)
            mo_k = -0.069 + 0.66 * (1.25 - 0.0042 * (d_k - 50.0))
            si_k = -2.868 * math.exp(-0.085 * WS[i]) + 1.0 + mo_k
            gamma_k = _fmax(0.0, si_k * math.exp(-z_i_k / 0.1))
            Dens_drift[k, i] = 0.0
            if si_k > 0.0 and subD[k, i] < Dfirn:
                tau_i_k = tau_drift / gamma_k
                drift_inc = dt_seconds * _fmax(350.0 - subD[k, i], 0.0) / tau_i_k
                subD[k, i] = _fmin(subD[k, i] + drift_inc, Dice)
                Dens_drift[k, i] = drift_inc
            z_i_k += subZ[k, i] * (3.25 - si_k)

    # ------ 3. UPDATE LAYER THICKNESS & SURFACE HEIGHT ------ #
    z_sum = 0.0
    z_sum_old = 0.0
    subW_sum = 0.0
    for k in range(nl):
        # subZ is only read in sections 1-2, so subZ[k, i] still holds the
        # pre-compaction thickness here. Take it before overwriting it below;
        # no separate subZ_old array is needed.
        z_old_k = subZ[k, i]
        if subD[k, i] < Dice:
            subZ[k, i] = z_old_k * subD_old[k, i] / subD[k, i]
            exp_f = 0.0143 * math.exp(3.3 * (Dice - subD[k, i]) / Dice)
            denom = 1.0 - exp_f
            mliqmax_k = subD[k, i] * subZ[k, i] * exp_f / denom * 0.05 * _fmin(Dice - subD[k, i], 20.0)
            if subW[k, i] > mliqmax_k:
                subW[k, i] = mliqmax_k
        else:
            subW[k, i] = 0.0
        z_sum += subZ[k, i]
        z_sum_old += z_old_k
        subW_sum += subW[k, i]

    surfH[i] += z_sum - z_sum_old
    runoff_irr[i] = sumWinit[i] - subW_sum


@cuda.jit
def _heat_conduction_kernel_gpu(
    subT,
    Tsurf,
    kk_sz_top,
    kk_sz_interior,
    dz1,
    dz2,
    denom_layer1,
    denom_interior,
    denom_bottom,
    dt_stab,
    dt,
    dayseconds,
    geothermal_flux,
    T_loc_ws,  # (gpsum, nl) float64 workspace — thread-local temperature copy
    kdTdz_ws,  # (gpsum, nl) float64 workspace — heat flux scratch
):
    """GPU equivalent of _heat_conduction_kernel. One CUDA thread per grid column."""
    i = cuda.grid(1)
    if i >= subT.shape[1]:
        return
    nl = subT.shape[0]

    for k in range(nl):
        T_loc_ws[k, i] = subT[k, i]
        kdTdz_ws[k, i] = 0.0

    tt_i = 0.0
    while tt_i < dt:
        dt_temp_i = _fmin(dt_stab[i], dt - tt_i)
        if dt_temp_i == 0.0:
            break
        tt_i += dt_temp_i
        C_day_dt = dayseconds * dt_temp_i

        # Freeze fluxes from current T_loc
        kdTdz_ws[1, i] = kk_sz_top[i] * (T_loc_ws[1, i] - Tsurf[i]) / dz1[i]
        for k in range(2, nl):
            kdTdz_ws[k, i] = kk_sz_interior[k - 2, i] * (T_loc_ws[k, i] - T_loc_ws[k - 1, i]) / dz2[k - 2, i]

        # Update T_loc in-place
        T_loc_ws[1, i] += C_day_dt * (kdTdz_ws[2, i] - kdTdz_ws[1, i]) / denom_layer1[i]
        for k in range(2, nl - 1):
            T_loc_ws[k, i] += C_day_dt * (kdTdz_ws[k + 1, i] - kdTdz_ws[k, i]) / denom_interior[k - 2, i]
        T_loc_ws[nl - 1, i] += C_day_dt * (geothermal_flux - kdTdz_ws[nl - 1, i]) / denom_bottom[i]

    for k in range(nl):
        subT[k, i] = T_loc_ws[k, i]


@cuda.jit
def _percolation_kernel_gpu(
    subT,
    subD,
    subW,
    subS,
    subZ,
    avail_W,
    RP,
    runoff_surface,
    runoff_slush,
    refr_P,
    refr_S,
    refr_I,
    slushw,
    irrw,
    T0,
    Dice,
    Dwater,
    Lm,
    Trunoff,
    perc_depth,
    percolation_mode,
    dt,
    wlim_ws,  # (gpsum, nl) float64 workspace
    wirr_ws,  # (gpsum, nl) float64 workspace
    carrot_ws,  # (gpsum, nl) float64 workspace
    slushspace_ws,  # (gpsum, nl) float64 workspace
):
    """GPU equivalent of _percolation_kernel. One CUDA thread per grid column."""
    i = cuda.grid(1)
    if i >= subT.shape[1]:
        return
    nl = subT.shape[0]

    sigma2_2 = 2.0 * (perc_depth / 3.0) ** 2
    norm_coeff = 2.0 / (perc_depth / 3.0) / math.sqrt(2.0 * math.pi)
    trunoff_factor = 1.0 / (1.0 + dt / Trunoff)

    # ------ Refreezing and Irreducible Water Storage Limits ------ #
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[k, i]
        c1_k = cpi_k * subD[k, i] * subZ[k, i] * (T0 - subT[k, i]) / Lm
        c2_k = subZ[k, i] * (1.0 - subD[k, i] / Dice) * Dice
        wlim_ws[k, i] = _fmax(_fmin(c1_k, c2_k), 0.0)
        if subD[k, i] < Dice - 1.0:
            factor_k = 3.3 * (Dice - subD[k, i]) / Dice
            exp_f = math.exp(factor_k)
            irr_f = 0.0143 * exp_f / (1.0 - 0.0143 * exp_f)
            mliqmax_k = subD[k, i] * subZ[k, i] * irr_f * 0.05 * _fmin(Dice - subD[k, i], 20.0)
        else:
            mliqmax_k = 0.0
        wirr_ws[k, i] = mliqmax_k - subW[k, i]

    # ------ Carrot (water-distribution profile) ------ #
    if percolation_mode == 0:
        carrot_ws[0, i] = 1.0
        for k in range(1, nl):
            carrot_ws[k, i] = 0.0
    else:
        depth = 0.0
        for k in range(nl):
            zz_k = depth + 0.5 * subZ[k, i]
            if percolation_mode == 1:
                carrot_ws[k, i] = norm_coeff * math.exp(-(zz_k * zz_k) / sigma2_2)
            elif percolation_mode == 2:
                v = 2.0 * (perc_depth - zz_k) / (perc_depth * perc_depth)
                carrot_ws[k, i] = v if v > 0.0 else 0.0
            else:
                carrot_ws[k, i] = zz_k
            depth += subZ[k, i]

        if percolation_mode == 3:
            min_dist = math.inf
            ind = 0
            for k in range(nl):
                d = _fabs(carrot_ws[k, i] - perc_depth)
                if d < min_dist:
                    min_dist = d
                    ind = k
            for k in range(nl):
                carrot_ws[k, i] = (1.0 / perc_depth) if k <= ind else 0.0

    s = 0.0
    for k in range(nl):
        carrot_ws[k, i] *= subZ[k, i]
        s += carrot_ws[k, i]
    avail_W_i = avail_W[i]
    for k in range(nl):
        carrot_ws[k, i] = carrot_ws[k, i] / s * avail_W_i

    # ------ Percolation: refreezing + irreducible storage ------ #
    avail_W_loc = 0.0
    rp_sum = 0.0
    for n in range(nl):
        avail_W_loc += carrot_ws[n, i]
        rp_n = _fmin(avail_W_loc, wlim_ws[n, i])
        RP[n, i] = rp_n
        excess = avail_W_loc - wlim_ws[n, i]
        if excess < 0.0:
            excess = 0.0
        # subW[n, i] still holds the pre-percolation value at this point; read
        # it before overwriting, so no separate subW_old array is needed.
        w_old_n = subW[n, i]
        new_subW_n = w_old_n + _fmin(excess, wirr_ws[n, i])
        subW[n, i] = new_subW_n
        avail_W_loc -= rp_n + (new_subW_n - w_old_n)
        cpi_n = 152.2 + 7.122 * subT[n, i]
        subT[n, i] += Lm * rp_n / (subD[n, i] * cpi_n * subZ[n, i])
        subD[n, i] += rp_n / subZ[n, i]
        rp_sum += rp_n

    avail_W[i] = avail_W_loc

    # ------ Slush water storage ------ #
    total_slushspace = 0.0
    for k in range(nl):
        ss_k = subZ[k, i] * (1.0 - subD[k, i] / Dice) * Dwater - subW[k, i]
        if ss_k < 0.0:
            ss_k = 0.0
        slushspace_ws[k, i] = ss_k
        total_slushspace += ss_k

    old_slush_sum = 0.0
    for k in range(nl):
        old_slush_sum += subS[k, i]
    avail_W_slush = avail_W_loc + old_slush_sum

    surf_ro = avail_W_slush - total_slushspace
    runoff_surface[i] = surf_ro if surf_ro > 0.0 else 0.0
    avail_S = avail_W_slush if avail_W_slush < total_slushspace else total_slushspace
    runoff_slush[i] = avail_S - trunoff_factor * avail_S
    avail_S = trunoff_factor * avail_S
    if avail_S < 1e-25:
        avail_S = 0.0

    for n in range(nl - 1, -1, -1):
        fill = avail_S if avail_S < slushspace_ws[n, i] else slushspace_ws[n, i]
        subS[n, i] = fill
        avail_S -= fill

    # ------ Slush refreezing ------ #
    rs_sum = 0.0
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[k, i]
        c1_k = cpi_k * subD[k, i] * subZ[k, i] * (T0 - subT[k, i]) / Lm
        c2_k = subZ[k, i] * (1.0 - subD[k, i] / Dice) * Dice
        wlim_k = _fmin(c1_k, c2_k)
        rs_k = 0.0
        if subS[k, i] > 0.0 and subT[k, i] < T0:
            rs_k = subS[k, i] if subS[k, i] < wlim_k else wlim_k
            if rs_k < 0.0:
                rs_k = 0.0
        subS[k, i] -= rs_k
        subT[k, i] += (Lm * rs_k) / (subD[k, i] * cpi_k * subZ[k, i])
        subD[k, i] += rs_k / subZ[k, i]
        rs_sum += rs_k

    # ------ Irreducible water refreezing ------ #
    ri_sum = 0.0
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[k, i]
        c1_k = cpi_k * subD[k, i] * subZ[k, i] * (T0 - subT[k, i]) / Lm
        c2_k = subZ[k, i] * (1.0 - subD[k, i] / Dice) * Dice
        wlim_k = _fmin(c1_k, c2_k)
        ri_k = 0.0
        if subW[k, i] > 0.0 and subT[k, i] < T0:
            ri_k = subW[k, i] if subW[k, i] < wlim_k else wlim_k
            if ri_k < 0.0:
                ri_k = 0.0
        subW[k, i] -= ri_k
        subT[k, i] += (Lm * ri_k) / (subD[k, i] * cpi_k * subZ[k, i])
        subD[k, i] += ri_k / subZ[k, i]
        ri_sum += ri_k

    slushw_i = 0.0
    irrw_i = 0.0
    for k in range(nl):
        slushw_i += subS[k, i]
        irrw_i += subW[k, i]
    refr_P[i] = 1e-3 * rp_sum
    refr_S[i] = 1e-3 * rs_sum
    refr_I[i] = 1e-3 * ri_sum
    slushw[i] = slushw_i
    irrw[i] = irrw_i


@cuda.jit
def _heat_conduction_prep_kernel_gpu(
    subD,  # (gpsum, nl)
    subZ,  # (gpsum, nl)
    subT,  # (gpsum, nl)
    kk,  # (gpsum, nl)      out: effective conductivity (scratch)
    c_eff,  # (gpsum, nl)      out: volumetric heat capacity (scratch)
    kk_sz_top,  # (gpsum,)         out
    kk_sz_interior,  # (gpsum, nl-2)    out
    dz1,  # (gpsum,)         out
    dz2,  # (gpsum, nl-2)    out
    denom_layer1,  # (gpsum,)         out
    denom_interior,  # (gpsum, nl-3)    out
    denom_bottom,  # (gpsum,)         out
    dt_stab,  # (gpsum,)         out
    dayseconds,
):
    """On-device heat-conduction precompute. One CUDA thread per grid column.

    Replaces the NumPy precompute block in heat_conduction() so that the
    derived arrays are produced directly from the resident (post-compaction)
    subD/subZ/subT without a host round-trip. Every formula is matched exactly
    to the NumPy path in LOOP_SNOW.heat_conduction() for --dump-reference parity.
    """
    i = cuda.grid(1)
    if i >= subD.shape[1]:
        return
    nl = subD.shape[0]

    # Effective conductivity and volumetric heat capacity per layer.
    for k in range(nl):
        d = subD[k, i]
        kk[k, i] = 0.138 - 1.01e-3 * d + 3.233e-6 * d * d
        c_eff[k, i] = d * (152.2 + 7.122 * subT[k, i])

    # dz1 = (subZ[0] + 0.5*subZ[1])**2
    half1 = subZ[0, i] + 0.5 * subZ[1, i]
    dz1[i] = half1 * half1

    # dz2 = 0.5 * (subZ[k+2] + subZ[k+1])**2   (NumPy: 0.5*(a+b)**2, NOT (0.5*(a+b))**2)
    for k in range(nl - 2):
        s = subZ[k + 2, i] + subZ[k + 1, i]
        dz2[k, i] = 0.5 * s * s

    # kk_sz_top = kk[0]*subZ[0] + 0.5*kk[1]*subZ[1]
    kk_sz_top[i] = kk[0, i] * subZ[0, i] + 0.5 * kk[1, i] * subZ[1, i]

    # kk_sz_interior[j] = kk[j+1]*subZ[j+1] + kk[j+2]*subZ[j+2]
    for k in range(nl - 2):
        kk_sz_interior[k, i] = kk[k + 1, i] * subZ[k + 1, i] + kk[k + 2, i] * subZ[k + 2, i]

    # denom_layer1 = c_eff[1] * (0.5*subZ[0] + 0.5*subZ[1] + 0.25*subZ[2])
    denom_layer1[i] = c_eff[1, i] * (0.5 * subZ[0, i] + 0.5 * subZ[1, i] + 0.25 * subZ[2, i])

    # denom_interior[k-2] = c_eff[k] * (0.25*subZ[k-1] + 0.5*subZ[k] + 0.25*subZ[k+1]) for k in 2..nl-2
    for k in range(2, nl - 1):
        denom_interior[k - 2, i] = c_eff[k, i] * (0.25 * subZ[k - 1, i] + 0.5 * subZ[k, i] + 0.25 * subZ[k + 1, i])

    # denom_bottom = c_eff[-1] * (0.25*subZ[-2] + 0.75*subZ[-1])
    denom_bottom[i] = c_eff[nl - 1, i] * (0.25 * subZ[nl - 2, i] + 0.75 * subZ[nl - 1, i])

    # dt_stab = 0.5 * min(c_eff[1:]) * min(subZ[1:])**2 / max(kk[1:]) / dayseconds
    min_ceff = math.inf
    min_sz = math.inf
    max_kk = 0.0
    for k in range(1, nl):
        if c_eff[k, i] < min_ceff:
            min_ceff = c_eff[k, i]
        if subZ[k, i] < min_sz:
            min_sz = subZ[k, i]
        if kk[k, i] > max_kk:
            max_kk = kk[k, i]
    dt_stab[i] = 0.5 * min_ceff * (min_sz * min_sz) / max_kk / dayseconds


@cuda.jit
def _heat_boundary_clip_kernel_gpu(subT, Tsurf, subZ, T0):
    """Surface ghost-layer update + melting-point clip. One thread per column.

    Matches the NumPy post-processing at the end of heat_conduction():
        subT[:,0] = Tsurf + (subT[:,1]-Tsurf)/(subZ[:,0]+0.5*subZ[:,1]) * 0.5*subZ[:,0]
        np.clip(subT, None, T0, out=subT)
    """
    i = cuda.grid(1)
    if i >= subT.shape[1]:
        return
    nl = subT.shape[0]

    subT[0, i] = Tsurf[i] + (subT[1, i] - Tsurf[i]) / (subZ[0, i] + 0.5 * subZ[1, i]) * 0.5 * subZ[0, i]
    for k in range(nl):
        if subT[k, i] > T0:
            subT[k, i] = T0


@cuda.jit
def _to_layer_major_kernel_gpu(src, dst):
    """(gpsum, nl) -> (nl, gpsum). One thread per column."""
    i = cuda.grid(1)
    if i >= src.shape[0]:
        return
    for k in range(src.shape[1]):
        dst[k, i] = src[i, k]


@cuda.jit
def _to_grid_major_kernel_gpu(src, dst):
    """(nl, gpsum) -> (gpsum, nl). One thread per column."""
    i = cuda.grid(1)
    if i >= dst.shape[0]:
        return
    for k in range(dst.shape[1]):
        dst[i, k] = src[k, i]


@cuda.jit
def _zero_kernel_gpu(arr):
    """Zero a (nl, gpsum) array in place. One thread per column."""
    i = cuda.grid(1)
    if i >= arr.shape[1]:
        return
    for k in range(arr.shape[0]):
        arr[k, i] = 0.0


# ===========================================================================
# Device-resident subsurface state.
#
# SnowDeviceState keeps the subsurface arrays (subT/subD/subZ/subW/subS/
# subTmean/surfH) on the GPU for the whole run. All six compute sub-steps of
# LOOP_SNOW -- snowfall/deposition, melt/sublimation, compaction, heat
# conduction, percolation, layer merging/splitting -- run against those
# buffers, so the grids are uploaded once at the start and only come back when
# host code actually needs them.
#
# Per timestep the transfers are:
#   H2D  the (gpsum,) forcing vectors (T, WS, snow, Tsurf, melt, moisture
#        fluxes, logyearsnow/yearsnow columns, percolation water input)
#   D2H  the (gpsum,) result vectors plus five (gpsum,) layer slices that host
#        code reads before the next LOOP_SNOW call -- see download_boundary()
#
# The full (gpsum, nl) grids move only on demand, via download_state(), for
# netCDF samples, the restart file and --dump-reference.
#
# The host-side physics glue (mode selection, water-input formula, unit
# conventions, output bookkeeping) stays in LOOP_SNOW.py, which is the single
# source of truth shared with the NumPy and Numba paths. This class only owns
# the device buffers and launches the kernels above against them, so
# --dump-reference parity with the NumPy/Numba paths is preserved.
# ===========================================================================


class SnowDeviceState:
    """GPU-resident subsurface state for LOOP_SNOW.

    Allocates all device buffers once for a given grid shape. upload_state()
    and upload_grid() run once per run; upload_forcing() refreshes this
    timestep's (gpsum,) inputs; the six sub-step methods launch their kernels
    against the resident buffers; download_boundary() returns the narrow slice
    the host reads between timesteps and download_state() the full grids.

    The instance is reused across timesteps (see get_device_state), so both the
    device allocations and the state upload happen once per run.
    """

    def __init__(self, gpsum, nl):
        self.shape = (gpsum, nl)
        self.blocks = _blocks(gpsum)

        # Host-layout staging buffer. Transfers keep the host's (gpsum, nl)
        # layout; a transpose kernel converts to/from the layer-major (nl,
        # gpsum) layout the compute kernels use, so consecutive threads read
        # consecutive addresses. Transposing on the device costs a single
        # extra pass and avoids a per-step host-side transpose.
        self._stage = cuda.device_array((gpsum, nl), dtype=np.float64)
        # Staging buffer for single-layer downloads. In the layer-major layout
        # one layer is a contiguous (gpsum,) row, so the host boundary slices
        # come back without a transpose.
        self._stage_col = np.empty(gpsum, dtype=np.float64)
        # The subsurface state is uploaded once per run and then stays on the
        # device across timesteps; see upload_state().
        self._state_uploaded = False
        self._grid_uploaded = False
        # Percolation water input, refreshed each step (was re-allocated with
        # cuda.to_device on every call).
        self._avail_W = cuda.device_array((gpsum,), dtype=np.float64)

        # Single-int32 device flag used to reproduce the host `np.any(...)`
        # conditions of the two grid-shift loops without downloading the
        # per-column shift arrays. See snowfall_and_deposition() /
        # melt_sublimation().
        self._flag = cuda.device_array((1,), dtype=np.int32)
        self._flag_host = np.zeros(1, dtype=np.int32)

        # Resident state: uploaded once per run (upload_state), read/written by
        # every sub-step kernel, copied back only via download_boundary() /
        # download_state().
        self.subT = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.subD = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.subZ = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.subW = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.subS = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.subTmean = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.surfH = cuda.device_array((gpsum,), dtype=np.float64)

        # Per-step forcing (read-only on the device; see upload_forcing).
        # logyearsnow / yearsnow are np.tile(x[:, None], (1, nl)) on the host,
        # i.e. identical for every layer. Keep only the (gpsum,) vector on the
        # device: 3 MB per step instead of 149 MB.
        self.logyearsnow = cuda.device_array((gpsum,), dtype=np.float64)
        self.yearsnow = cuda.device_array((gpsum,), dtype=np.float64)
        self.WS = cuda.device_array((gpsum,), dtype=np.float64)
        self.Tsurf = cuda.device_array((gpsum,), dtype=np.float64)
        self.T = cuda.device_array((gpsum,), dtype=np.float64)
        self.snow = cuda.device_array((gpsum,), dtype=np.float64)
        self.melt = cuda.device_array((gpsum,), dtype=np.float64)
        self.moist_deposition = cuda.device_array((gpsum,), dtype=np.float64)
        self.moist_sublimation = cuda.device_array((gpsum,), dtype=np.float64)

        # Grid geometry for layer merging/splitting; uploaded once (see
        # upload_grid).
        self.mask = cuda.device_array((gpsum,), dtype=np.int32)
        self.split = None

        # Produced on the device, consumed on the device.
        self.sumWinit = cuda.device_array((gpsum,), dtype=np.float64)
        self.shift_tot = cuda.device_array((gpsum,), dtype=np.float64)
        self.Dfreshsnow = cuda.device_array((gpsum,), dtype=np.float64)
        self.Dfreshsnow_T = cuda.device_array((gpsum,), dtype=np.float64)
        self.Dfreshsnow_W = cuda.device_array((gpsum,), dtype=np.float64)
        self.runoff_irr_deep = cuda.device_array((gpsum,), dtype=np.float64)
        # np.all(subD >= Dice, axis=1) for LOOP_mass_balance, so the host never
        # has to reduce over the resident density grid.
        self.all_ice = cuda.device_array((gpsum,), dtype=np.uint8)

        # Dens_* diagnostics, zeroed on the device at the start of compaction.
        self.Dens_destr_metam = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.Dens_overb_pres = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.Dens_drift = cuda.device_array((nl, gpsum), dtype=np.float64)

        # Conductivity / heat capacity filled by the heat-conduction prep kernel
        # and kept resident so download_state() returns subK/subCeff matching NumPy.
        self.kk = cuda.device_array((nl, gpsum), dtype=np.float64)
        self.c_eff = cuda.device_array((nl, gpsum), dtype=np.float64)

        # Write-only column outputs (allocated, never uploaded).
        self.runoff_irr = cuda.device_array((gpsum,), dtype=np.float64)
        self.runoff_surface = cuda.device_array((gpsum,), dtype=np.float64)
        self.runoff_slush = cuda.device_array((gpsum,), dtype=np.float64)
        self.refr_P = cuda.device_array((gpsum,), dtype=np.float64)
        self.refr_S = cuda.device_array((gpsum,), dtype=np.float64)
        self.refr_I = cuda.device_array((gpsum,), dtype=np.float64)
        self.slushw = cuda.device_array((gpsum,), dtype=np.float64)
        self.irrw = cuda.device_array((gpsum,), dtype=np.float64)

        # Device-only scratch (never touches the host).
        self._subD_old = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._was_snow_ws = cuda.device_array((nl, gpsum), dtype=np.bool_)
        self._psload_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._kk_sz_top = cuda.device_array((gpsum,), dtype=np.float64)
        self._kk_sz_interior = cuda.device_array((nl - 2, gpsum), dtype=np.float64)
        self._dz1 = cuda.device_array((gpsum,), dtype=np.float64)
        self._dz2 = cuda.device_array((nl - 2, gpsum), dtype=np.float64)
        self._denom_layer1 = cuda.device_array((gpsum,), dtype=np.float64)
        self._denom_interior = cuda.device_array((nl - 3, gpsum), dtype=np.float64)
        self._denom_bottom = cuda.device_array((gpsum,), dtype=np.float64)
        self._dt_stab = cuda.device_array((gpsum,), dtype=np.float64)
        self._T_loc_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._kdTdz_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._RP = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._wlim_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._wirr_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._carrot_ws = cuda.device_array((nl, gpsum), dtype=np.float64)
        self._slushspace_ws = cuda.device_array((nl, gpsum), dtype=np.float64)

    def upload_state(self, OUT):
        """Upload the subsurface state -- once per run, not once per timestep.

        subT/subD/subZ/subW/subS/subTmean/surfH are written only by LOOP_SNOW,
        which now runs entirely on the device, so after this initial upload the
        device copy is the authoritative one for the rest of the run. Uploading
        again would overwrite it with whatever stale values the host arrays
        happen to hold.
        """
        if self._state_uploaded:
            return
        self._upload_2d(self.subT, OUT["subT"])
        self._upload_2d(self.subD, OUT["subD"])
        self._upload_2d(self.subZ, OUT["subZ"])
        self._upload_2d(self.subW, OUT["subW"])
        self._upload_2d(self.subS, OUT["subS"])
        self._upload_2d(self.subTmean, OUT["subTmean"])
        self.surfH.copy_to_device(OUT["surfH"])
        self._state_uploaded = True

    def upload_grid(self, grid):
        """Upload the layer-merging geometry -- once per run; it never changes."""
        if self._grid_uploaded:
            return
        self.mask.copy_to_device(np.ascontiguousarray(grid["mask"], dtype=np.int32))
        self.split = cuda.to_device(np.ascontiguousarray(grid["split"], dtype=np.int32))
        self._grid_uploaded = True

    def upload_forcing(self, OUT, IN):
        """Upload this timestep's forcing. All (gpsum,) vectors, no grids."""
        # logyearsnow / yearsnow are np.tile(x[:, None], (1, nl)) on the host,
        # i.e. layer-invariant; upload the column, not the grid.
        self.logyearsnow.copy_to_device(np.ascontiguousarray(IN["logyearsnow"][:, 0]))
        self.yearsnow.copy_to_device(np.ascontiguousarray(IN["yearsnow"][:, 0]))
        self.T.copy_to_device(np.ascontiguousarray(IN["T"]))
        self.WS.copy_to_device(np.ascontiguousarray(IN["WS"]))
        self.snow.copy_to_device(np.ascontiguousarray(IN["snow"]))
        self.Tsurf.copy_to_device(np.ascontiguousarray(OUT["Tsurf"]))
        self.melt.copy_to_device(np.ascontiguousarray(OUT["melt"]))
        self.moist_deposition.copy_to_device(np.ascontiguousarray(OUT["moist_deposition"]))
        self.moist_sublimation.copy_to_device(np.ascontiguousarray(OUT["moist_sublimation"]))

    def _upload_2d(self, dst, host_arr):
        """Host (gpsum, nl) -> device (nl, gpsum) via the staging buffer."""
        self._stage.copy_to_device(host_arr)
        _to_layer_major_kernel_gpu[self.blocks, _TPB](self._stage, dst)

    def _download_2d(self, src, host_arr):
        """Device (nl, gpsum) -> host (gpsum, nl) via the staging buffer."""
        _to_grid_major_kernel_gpu[self.blocks, _TPB](src, self._stage)
        self._stage.copy_to_host(host_arr)

    def _reset_flag(self):
        """Clear the device 'more work to do' flag before a shift-loop pass."""
        self._flag_host[0] = 0
        self._flag.copy_to_device(self._flag_host)

    def _flag_is_set(self) -> bool:
        """Read back the device flag, i.e. the host's `np.any(...)` condition."""
        self._flag.copy_to_host(self._flag_host)
        return bool(self._flag_host[0])

    def snowfall_and_deposition(self, C, grid, mode):
        """Launch the snowfall / deposition kernels on the resident arrays.

        The host `while np.any(shift_tot > 0)` loop becomes a host-driven
        kernel relaunch: the kernels raise a single int32 device flag, which is
        the only thing copied back per pass (4 bytes). Driving the loop from
        the host rather than per thread matters for bit-exactness -- the NumPy
        path applies the loop body to *every* column on every pass, including
        the ones that are already finished, and that body is not an exact no-op
        in floating point.
        """
        self._reset_flag()
        _snowfall_prep_kernel_gpu[self.blocks, _TPB](
            self.subZ,
            self.Dfreshsnow,
            self.Dfreshsnow_T,
            self.Dfreshsnow_W,
            self.shift_tot,
            self.surfH,
            self.runoff_irr_deep,
            self.T,
            self.WS,
            self.snow,
            self.moist_deposition,
            C["Dwater"],
            C["T0"],
            C["Dfreshsnow"],
            mode,
            self._flag,
        )
        while self._flag_is_set():
            self._reset_flag()
            _snowfall_shift_kernel_gpu[self.blocks, _TPB](
                self.subZ,
                self.subT,
                self.subD,
                self.subW,
                self.Dfreshsnow,
                self.Tsurf,
                self.shift_tot,
                self.runoff_irr_deep,
                grid["max_subZ"],
                self._flag,
            )

    def melt_sublimation(self, bottom_thickness):
        """Launch the melt / sublimation kernels on the resident arrays.

        Same host-driven pattern as snowfall_and_deposition() for the grid
        shift. The preceding layer-peeling loop is genuinely a no-op for
        finished columns (both of its NumPy masks are False there), so it runs
        per thread with a private trip count and needs no flag round-trip.
        """
        self._reset_flag()
        _melt_peel_kernel_gpu[self.blocks, _TPB](
            self.subZ,
            self.subD,
            self.subW,
            self.melt,
            self.moist_sublimation,
            self.sumWinit,
            self.shift_tot,
            self._flag,
        )
        while self._flag_is_set():
            self._reset_flag()
            _melt_shift_kernel_gpu[self.blocks, _TPB](
                self.subZ,
                self.subT,
                self.subD,
                self.subW,
                self.surfH,
                self.shift_tot,
                bottom_thickness,
                self._flag,
            )

    def layer_merging_and_splitting(self, max_subZ, top_thickness):
        """Launch the layer merging / splitting kernel on the resident arrays."""
        _layer_merging_splitting_kernel_gpu[self.blocks, _TPB](
            self.subZ,
            self.subT,
            self.subD,
            self.subW,
            self.subS,
            self.runoff_irr_deep,
            self.runoff_slush,
            self.mask,
            self.split,
            max_subZ,
            top_thickness,
        )

    def compaction(
        self,
        dt_yearfrac,
        dt_seconds,
        dt,
        C,
        tau_drift,
        mode,
    ):
        """Launch the compaction kernel on the resident arrays."""
        # The kernel only writes the Dens_* diagnostics in firn+snow mode, so
        # they must start at zero. Zero them on the device rather than uploading
        # host zeros (3 full grids per timestep of pure PCIe traffic).
        _zero_kernel_gpu[self.blocks, _TPB](self.Dens_destr_metam)
        _zero_kernel_gpu[self.blocks, _TPB](self.Dens_overb_pres)
        _zero_kernel_gpu[self.blocks, _TPB](self.Dens_drift)

        # Snapshot pre-compaction subD/subZ on-device (host path does .copy()).
        self._subD_old.copy_to_device(self.subD)

        _compaction_kernel_gpu[self.blocks, _TPB](
            self.subD,
            self.subZ,
            self.subT,
            self.subW,
            self.subTmean,
            self._subD_old,
            self.logyearsnow,
            self.yearsnow,
            self.WS,
            self.Dens_destr_metam,
            self.Dens_overb_pres,
            self.Dens_drift,
            self.surfH,
            self.sumWinit,
            self.runoff_irr,
            dt_yearfrac,
            dt_seconds,
            dt,
            C["Dice"],
            C["Dfirn"],
            C["Dwater"],
            C["g"],
            C["T0"],
            C["rd"],
            C["Ec"],
            C["Eg"],
            C["dayseconds"],
            tau_drift,
            mode,
            self._was_snow_ws,
            self._psload_ws,
        )

    def heat_conduction(self, dt, C):
        """Launch heat-conduction prep, solve and boundary/clip, all on-device."""
        _heat_conduction_prep_kernel_gpu[self.blocks, _TPB](
            self.subD,
            self.subZ,
            self.subT,
            self.kk,
            self.c_eff,
            self._kk_sz_top,
            self._kk_sz_interior,
            self._dz1,
            self._dz2,
            self._denom_layer1,
            self._denom_interior,
            self._denom_bottom,
            self._dt_stab,
            C["dayseconds"],
        )
        _heat_conduction_kernel_gpu[self.blocks, _TPB](
            self.subT,
            self.Tsurf,
            self._kk_sz_top,
            self._kk_sz_interior,
            self._dz1,
            self._dz2,
            self._denom_layer1,
            self._denom_interior,
            self._denom_bottom,
            self._dt_stab,
            dt,
            C["dayseconds"],
            C["geothermal_flux"],
            self._T_loc_ws,
            self._kdTdz_ws,
        )
        _heat_boundary_clip_kernel_gpu[self.blocks, _TPB](
            self.subT,
            self.Tsurf,
            self.subZ,
            C["T0"],
        )

    def percolation(self, avail_W, C, perc_mode, dt):
        """Launch the percolation kernel on the resident arrays."""
        self._avail_W.copy_to_device(avail_W)
        # Snapshot post-heat subW on-device (host path does subW.copy()).
        _percolation_kernel_gpu[self.blocks, _TPB](
            self.subT,
            self.subD,
            self.subW,
            self.subS,
            self.subZ,
            self._avail_W,
            self._RP,
            self.runoff_surface,
            self.runoff_slush,
            self.refr_P,
            self.refr_S,
            self.refr_I,
            self.slushw,
            self.irrw,
            C["T0"],
            C["Dice"],
            C["Dwater"],
            C["Lm"],
            C["Trunoff"],
            C["perc_depth"],
            perc_mode,
            dt,
            self._wlim_ws,
            self._wirr_ws,
            self._carrot_ws,
            self._slushspace_ws,
        )

    def _download_layer(self, src, k, host_grid, col):
        """Copy device layer ``k`` into column ``col`` of a host (gpsum, nl) grid.

        In the layer-major device layout a layer is one contiguous (gpsum,)
        row, so this is a single coalesced copy with no transpose.
        """
        src[k].copy_to_host(self._stage_col)
        host_grid[:, col] = self._stage_col

    def download_boundary(self, OUT, Dice):
        """Copy back only what host code reads before the next LOOP_SNOW call.

        With every LOOP_SNOW sub-step on the device, the subsurface grids no
        longer have to round-trip per timestep. Between two LOOP_SNOW calls the
        host reads exactly:

            LOOP_EBM          subD[:, :2], subZ[:, :2] (GHF coefficients)
            LOOP_EBM_GHF      subT[:, 1]
            LOOP_EBM_SWout    subD[:, 0]
            LOOP_mass_balance np.all(subD >= Dice, axis=1)   -> computed here
            LOOP_SNOW.main    subT[:, -1] (T_ice)
            runoff()          the (gpsum,) runoff / refreezing vectors

        so only those come back: five (gpsum,) layer slices plus the per-column
        vectors, instead of five full (gpsum, nl) grids.

        IMPORTANT: this leaves OUT["subT"], OUT["subD"], OUT["subZ"],
        OUT["subW"] and OUT["subS"] only *partially* refreshed -- the layers
        listed above are current, the rest are stale. Any host code that reads
        a full subsurface grid must call LOOP_SNOW.sync_gpu_state(OUT) first.
        """
        _all_ice_kernel_gpu[self.blocks, _TPB](self.subD, Dice, self.all_ice)

        nl = self.shape[1]
        self._download_layer(self.subD, 0, OUT["subD"], 0)
        self._download_layer(self.subD, 1, OUT["subD"], 1)
        self._download_layer(self.subZ, 0, OUT["subZ"], 0)
        self._download_layer(self.subZ, 1, OUT["subZ"], 1)
        self._download_layer(self.subT, 1, OUT["subT"], 1)

        # T_ice is the deepest layer; hand it over directly rather than through
        # the (otherwise stale) OUT["subT"] grid.
        self.subT[nl - 1].copy_to_host(self._stage_col)
        OUT["T_ice"] = self._stage_col.copy()

        self.surfH.copy_to_host(OUT["surfH"])
        self.runoff_irr.copy_to_host(OUT["runoff_irr"])
        OUT["runoff_irr_deep"] = self.runoff_irr_deep.copy_to_host()

        # Percolation outputs. refr_* already carry the 1e-3 factor from the
        # kernel; runoff_* stay in mm and are scaled later in runoff().
        OUT["runoff_surface"] = self.runoff_surface.copy_to_host()
        OUT["runoff_slush"] = self.runoff_slush.copy_to_host()
        OUT["refr_P"] = self.refr_P.copy_to_host()
        OUT["refr_S"] = self.refr_S.copy_to_host()
        OUT["refr_I"] = self.refr_I.copy_to_host()
        OUT["slushw"] = self.slushw.copy_to_host()
        OUT["irrw"] = self.irrw.copy_to_host()
        OUT["refr"] = OUT["refr_P"] + OUT["refr_S"] + OUT["refr_I"]

        # Consumed by LOOP_mass_balance in place of its own full-grid reduction.
        OUT["all_ice_column"] = self.all_ice.copy_to_host().astype(bool)

    def download_state(self, OUT):
        """Copy the full device-resident state and diagnostics back into OUT.

        Called only when host code actually needs the complete grids: writing a
        netCDF sample, a restart file, or a --dump-reference snapshot. Every
        array here is either write-only during the time loop (the Dens_*
        diagnostics, subK / subCeff, subTmean) or covered layer-wise by
        download_boundary(), so nothing between timesteps depends on it.
        """
        for src, key in (
            (self.subT, "subT"),
            (self.subD, "subD"),
            (self.subZ, "subZ"),
            (self.subW, "subW"),
            (self.subS, "subS"),
            (self.subTmean, "subTmean"),
            (self.Dens_destr_metam, "Dens_destr_metam"),
            (self.Dens_overb_pres, "Dens_overb_pres"),
            (self.Dens_drift, "Dens_drift"),
            (self.kk, "subK"),
            (self.c_eff, "subCeff"),
        ):
            host = OUT.get(key)
            if host is None or host.shape != self.shape:
                host = np.empty(self.shape, dtype=np.float64)
                OUT[key] = host
            self._download_2d(src, host)

        self.sumWinit.copy_to_host(OUT["sumWinit"])
        OUT["Dfreshsnow"] = self.Dfreshsnow.copy_to_host()
        OUT["Dfreshsnow_T"] = self.Dfreshsnow_T.copy_to_host()
        OUT["Dfreshsnow_W"] = self.Dfreshsnow_W.copy_to_host()
        OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]


# ---------------------------------------------------------------------------
# Cached device state
#
# The device buffers depend only on the grid shape, so one SnowDeviceState is
# allocated per run and reused for every timestep. Re-creating it each step
# would re-allocate ~25 device arrays, and cudaMalloc/cudaFree serialise
# against the compute stream.
#
# Module-level state mirrors how compute_backend.py keeps the active backend.
# ---------------------------------------------------------------------------

_device_state = None


def sync_device_state(OUT):
    """Copy the full device-resident state into OUT, if a GPU state exists."""
    if _device_state is not None:
        _device_state.download_state(OUT)


def get_device_state(OUT, IN, grid):
    """Return the cached SnowDeviceState, ready for this timestep.

    Allocates the device buffers and uploads the subsurface state on the first
    call (or if the grid shape changed); afterwards only this timestep's
    (gpsum,) forcing vectors are uploaded, because the subsurface state stays
    resident across timesteps.
    """
    global _device_state
    if _device_state is None or _device_state.shape != OUT["subD"].shape:
        gpsum, nl = OUT["subD"].shape
        _device_state = SnowDeviceState(gpsum, nl)
    _device_state.upload_state(OUT)
    _device_state.upload_grid(grid)
    _device_state.upload_forcing(OUT, IN)
    return _device_state


# ===========================================================================
# Smoke test
# ===========================================================================


def gpu_offload_smoke_test() -> dict:
    """Verify GPU availability by running a trivial round-trip kernel.

    Transfers a small array to the GPU, doubles each element, copies the result
    back, and asserts correctness. Also queries device name and memory.

    Returns
    -------
    dict
        ``{"available": True, "device_name": ..., "free_mem_gb": ...,
           "total_mem_gb": ...}`` on success, or
        ``{"available": False, "reason": ...}`` on failure.
    """
    try:
        devices = list(cuda.list_devices())
    except Exception as exc:
        return {"available": False, "reason": f"cuda.list_devices() failed: {exc}"}

    if not devices:
        return {"available": False, "reason": "no CUDA/ROCm devices detected"}

    device = devices[0]
    try:
        device_name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    except Exception:
        device_name = repr(device)

    try:
        with cuda.gpus[0]:
            free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception as exc:
        return {"available": False, "reason": f"failed to query device memory: {exc}"}

    # Round-trip test: allocate 32 ones, double on GPU, assert result == 2.
    host_arr = np.ones(32, dtype=np.float64)
    try:
        d_arr = cuda.to_device(host_arr)
        _gpu_probe_kernel[4, 8](d_arr)  # 4 blocks x 8 threads = 32 threads
        result = d_arr.copy_to_host()
    except Exception as exc:
        return {"available": False, "reason": f"GPU kernel launch failed: {exc}"}

    if not np.allclose(result, 2.0):
        return {
            "available": False,
            "reason": f"GPU result mismatch: expected 2.0, got max={result.max():.6g}",
        }

    return {
        "available": True,
        "device_name": device_name,
        "free_mem_gb": free_bytes / 2**30,
        "total_mem_gb": total_bytes / 2**30,
    }
