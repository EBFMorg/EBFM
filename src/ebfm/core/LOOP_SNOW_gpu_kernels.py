# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GPU kernels for the LOOP_SNOW module.

Availability: numba.cuda kernels for compaction, heat conduction and
percolation, offloaded to a GPU when the GPU backend is active (--with-gpu).

The same kernels run on both vendors:
- NVIDIA via stock numba.cuda,
- AMD via numba.hip.pose_as_cuda() (see compute_backend.py).

Each ``_*_kernel_gpu`` is the CUDA-thread equivalent of the corresponding
``@njit`` kernel in LOOP_SNOW_kernels.py: the outer ``prange(gpsum)`` loop is
replaced by a single CUDA thread index so one GPU thread handles one grid
column. Per-column scratch arrays (suffix ``_ws``, shape ``(gpsum, nl)``) are
pre-allocated on the device by ``SnowDeviceState`` to avoid dynamic allocation
inside the kernels.

``SnowDeviceState`` owns the device buffers for one LOOP_SNOW step: it uploads
the subsurface state once, launches the compaction / heat-conduction /
percolation kernels back-to-back while the shared arrays stay resident on the
device, and copies every result back in one shot. The host-side physics glue
lives in LOOP_SNOW.py (shared with the NumPy / Numba paths), so results match
those paths exactly.

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
def _compaction_kernel_gpu(
    subD,
    subZ,
    subT,
    subW,
    subTmean,
    subD_old,
    logyearsnow,
    yearsnow,
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
    if i >= subD.shape[0]:
        return
    nl = subD.shape[1]

    # ------ 1. FIRN COMPACTION ------ #
    for k in range(nl):
        subTmean[i, k] = subTmean[i, k] * (1.0 - dt_yearfrac) + dt_yearfrac * subT[i, k]
        cond_firn_k = (compaction_mode == 0) or (subD[i, k] >= Dfirn)
        if cond_firn_k:
            if subD[i, k] < 550.0:
                grav_const = 0.07 * _fmax(1.435 - 0.151 * logyearsnow[i, k], 0.25)
            else:
                grav_const = 0.03 * _fmax(2.366 - 0.293 * logyearsnow[i, k], 0.25)
            temp_factor = math.exp(-Ec / (rd * subT[i, k]) + Eg / (rd * subTmean[i, k]))
            firn_inc = dt_yearfrac * grav_const * yearsnow[i, k] * g * (Dice - subD[i, k]) * temp_factor
            subD[i, k] += firn_inc

    # ------ 2. SEASONAL SNOW COMPACTION ------ #
    if compaction_mode == 1:  # firn+snow
        # Capture pre-DM snow mask
        for k in range(nl):
            was_snow_ws[i, k] = subD[i, k] < Dfirn

        # ------ 2.1 DESTRUCTIVE METAMORPHISM ------ #
        for k in range(nl):
            if was_snow_ws[i, k]:
                cc1 = math.exp(-0.046 * _fmax(subD[i, k] - 175.0, 0.0))
                cc2 = 1.0 + (1.0 if subW[i, k] != 0.0 else 0.0)
                temp_exp = math.exp(0.04 * (subT[i, k] - T0))
                snow_inc = cc1 * cc2 * 2.777e-6 * temp_exp * dt_seconds * subD[i, k]
                subD[i, k] = _fmin(subD[i, k] + snow_inc, Dice)
                Dens_destr_metam[i, k] = snow_inc
            else:
                Dens_destr_metam[i, k] = 0.0

        # ------ 2.2 OVERBURDEN PRESSURE ------ #
        psload_ws[i, 0] = 0.5 * subD[i, 0] * subZ[i, 0] * g
        for k in range(1, nl):
            xm = subD[i, k - 1] * subZ[i, k - 1] * g
            xk = subD[i, k] * subZ[i, k] * g
            psload_ws[i, k] = psload_ws[i, k - 1] + 0.5 * (xm + xk)

        for k in range(nl):
            Dens_overb_pres[i, k] = 0.0
            if was_snow_ws[i, k]:
                cc7 = 4.0 * 7.62237e6 / 250.0 * subD[i, k] / (1.0 + 60.0 * subW[i, k] / (Dwater * subZ[i, k]))
                visc = cc7 * math.exp(0.1 * (T0 - subT[i, k]) + 0.023 * subD[i, k])
                overb_inc = dt * dayseconds * subD[i, k] * psload_ws[i, k] / visc
                subD[i, k] = _fmin(subD[i, k] + overb_inc, Dice)
                Dens_overb_pres[i, k] = dt * dayseconds * subD[i, k] * psload_ws[i, k] / visc

        # ------ 2.3 DRIFTING SNOW ------ #
        z_i_k = 0.0
        for k in range(nl):
            d_k = _fmax(subD[i, k], 50.0)
            mo_k = -0.069 + 0.66 * (1.25 - 0.0042 * (d_k - 50.0))
            si_k = -2.868 * math.exp(-0.085 * WS[i]) + 1.0 + mo_k
            gamma_k = _fmax(0.0, si_k * math.exp(-z_i_k / 0.1))
            Dens_drift[i, k] = 0.0
            if si_k > 0.0 and subD[i, k] < Dfirn:
                tau_i_k = tau_drift / gamma_k
                drift_inc = dt_seconds * _fmax(350.0 - subD[i, k], 0.0) / tau_i_k
                subD[i, k] = _fmin(subD[i, k] + drift_inc, Dice)
                Dens_drift[i, k] = drift_inc
            z_i_k += subZ[i, k] * (3.25 - si_k)

    # ------ 3. UPDATE LAYER THICKNESS & SURFACE HEIGHT ------ #
    z_sum = 0.0
    z_sum_old = 0.0
    subW_sum = 0.0
    for k in range(nl):
        # subZ is only read in sections 1-2, so subZ[i, k] still holds the
        # pre-compaction thickness here. Take it before overwriting it below;
        # no separate subZ_old array is needed.
        z_old_k = subZ[i, k]
        if subD[i, k] < Dice:
            subZ[i, k] = z_old_k * subD_old[i, k] / subD[i, k]
            exp_f = 0.0143 * math.exp(3.3 * (Dice - subD[i, k]) / Dice)
            denom = 1.0 - exp_f
            mliqmax_k = subD[i, k] * subZ[i, k] * exp_f / denom * 0.05 * _fmin(Dice - subD[i, k], 20.0)
            if subW[i, k] > mliqmax_k:
                subW[i, k] = mliqmax_k
        else:
            subW[i, k] = 0.0
        z_sum += subZ[i, k]
        z_sum_old += z_old_k
        subW_sum += subW[i, k]

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
    if i >= subT.shape[0]:
        return
    nl = subT.shape[1]

    for k in range(nl):
        T_loc_ws[i, k] = subT[i, k]
        kdTdz_ws[i, k] = 0.0

    tt_i = 0.0
    while tt_i < dt:
        dt_temp_i = _fmin(dt_stab[i], dt - tt_i)
        if dt_temp_i == 0.0:
            break
        tt_i += dt_temp_i
        C_day_dt = dayseconds * dt_temp_i

        # Freeze fluxes from current T_loc
        kdTdz_ws[i, 1] = kk_sz_top[i] * (T_loc_ws[i, 1] - Tsurf[i]) / dz1[i]
        for k in range(2, nl):
            kdTdz_ws[i, k] = kk_sz_interior[i, k - 2] * (T_loc_ws[i, k] - T_loc_ws[i, k - 1]) / dz2[i, k - 2]

        # Update T_loc in-place
        T_loc_ws[i, 1] += C_day_dt * (kdTdz_ws[i, 2] - kdTdz_ws[i, 1]) / denom_layer1[i]
        for k in range(2, nl - 1):
            T_loc_ws[i, k] += C_day_dt * (kdTdz_ws[i, k + 1] - kdTdz_ws[i, k]) / denom_interior[i, k - 2]
        T_loc_ws[i, nl - 1] += C_day_dt * (geothermal_flux - kdTdz_ws[i, nl - 1]) / denom_bottom[i]

    for k in range(nl):
        subT[i, k] = T_loc_ws[i, k]


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
    if i >= subT.shape[0]:
        return
    nl = subT.shape[1]

    sigma2_2 = 2.0 * (perc_depth / 3.0) ** 2
    norm_coeff = 2.0 / (perc_depth / 3.0) / math.sqrt(2.0 * math.pi)
    trunoff_factor = 1.0 / (1.0 + dt / Trunoff)

    # ------ Refreezing and Irreducible Water Storage Limits ------ #
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[i, k]
        c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
        c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
        wlim_ws[i, k] = _fmax(_fmin(c1_k, c2_k), 0.0)
        if subD[i, k] < Dice - 1.0:
            factor_k = 3.3 * (Dice - subD[i, k]) / Dice
            exp_f = math.exp(factor_k)
            irr_f = 0.0143 * exp_f / (1.0 - 0.0143 * exp_f)
            mliqmax_k = subD[i, k] * subZ[i, k] * irr_f * 0.05 * _fmin(Dice - subD[i, k], 20.0)
        else:
            mliqmax_k = 0.0
        wirr_ws[i, k] = mliqmax_k - subW[i, k]

    # ------ Carrot (water-distribution profile) ------ #
    if percolation_mode == 0:
        carrot_ws[i, 0] = 1.0
        for k in range(1, nl):
            carrot_ws[i, k] = 0.0
    else:
        depth = 0.0
        for k in range(nl):
            zz_k = depth + 0.5 * subZ[i, k]
            if percolation_mode == 1:
                carrot_ws[i, k] = norm_coeff * math.exp(-(zz_k * zz_k) / sigma2_2)
            elif percolation_mode == 2:
                v = 2.0 * (perc_depth - zz_k) / (perc_depth * perc_depth)
                carrot_ws[i, k] = v if v > 0.0 else 0.0
            else:
                carrot_ws[i, k] = zz_k
            depth += subZ[i, k]

        if percolation_mode == 3:
            min_dist = math.inf
            ind = 0
            for k in range(nl):
                d = _fabs(carrot_ws[i, k] - perc_depth)
                if d < min_dist:
                    min_dist = d
                    ind = k
            for k in range(nl):
                carrot_ws[i, k] = (1.0 / perc_depth) if k <= ind else 0.0

    s = 0.0
    for k in range(nl):
        carrot_ws[i, k] *= subZ[i, k]
        s += carrot_ws[i, k]
    avail_W_i = avail_W[i]
    for k in range(nl):
        carrot_ws[i, k] = carrot_ws[i, k] / s * avail_W_i

    # ------ Percolation: refreezing + irreducible storage ------ #
    avail_W_loc = 0.0
    rp_sum = 0.0
    for n in range(nl):
        avail_W_loc += carrot_ws[i, n]
        rp_n = _fmin(avail_W_loc, wlim_ws[i, n])
        RP[i, n] = rp_n
        excess = avail_W_loc - wlim_ws[i, n]
        if excess < 0.0:
            excess = 0.0
        # subW[i, n] still holds the pre-percolation value at this point; read
        # it before overwriting, so no separate subW_old array is needed.
        w_old_n = subW[i, n]
        new_subW_n = w_old_n + _fmin(excess, wirr_ws[i, n])
        subW[i, n] = new_subW_n
        avail_W_loc -= rp_n + (new_subW_n - w_old_n)
        cpi_n = 152.2 + 7.122 * subT[i, n]
        subT[i, n] += Lm * rp_n / (subD[i, n] * cpi_n * subZ[i, n])
        subD[i, n] += rp_n / subZ[i, n]
        rp_sum += rp_n

    avail_W[i] = avail_W_loc

    # ------ Slush water storage ------ #
    total_slushspace = 0.0
    for k in range(nl):
        ss_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dwater - subW[i, k]
        if ss_k < 0.0:
            ss_k = 0.0
        slushspace_ws[i, k] = ss_k
        total_slushspace += ss_k

    old_slush_sum = 0.0
    for k in range(nl):
        old_slush_sum += subS[i, k]
    avail_W_slush = avail_W_loc + old_slush_sum

    surf_ro = avail_W_slush - total_slushspace
    runoff_surface[i] = surf_ro if surf_ro > 0.0 else 0.0
    avail_S = avail_W_slush if avail_W_slush < total_slushspace else total_slushspace
    runoff_slush[i] = avail_S - trunoff_factor * avail_S
    avail_S = trunoff_factor * avail_S
    if avail_S < 1e-25:
        avail_S = 0.0

    for n in range(nl - 1, -1, -1):
        fill = avail_S if avail_S < slushspace_ws[i, n] else slushspace_ws[i, n]
        subS[i, n] = fill
        avail_S -= fill

    # ------ Slush refreezing ------ #
    rs_sum = 0.0
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[i, k]
        c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
        c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
        wlim_k = _fmin(c1_k, c2_k)
        rs_k = 0.0
        if subS[i, k] > 0.0 and subT[i, k] < T0:
            rs_k = subS[i, k] if subS[i, k] < wlim_k else wlim_k
            if rs_k < 0.0:
                rs_k = 0.0
        subS[i, k] -= rs_k
        subT[i, k] += (Lm * rs_k) / (subD[i, k] * cpi_k * subZ[i, k])
        subD[i, k] += rs_k / subZ[i, k]
        rs_sum += rs_k

    # ------ Irreducible water refreezing ------ #
    ri_sum = 0.0
    for k in range(nl):
        cpi_k = 152.2 + 7.122 * subT[i, k]
        c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
        c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
        wlim_k = _fmin(c1_k, c2_k)
        ri_k = 0.0
        if subW[i, k] > 0.0 and subT[i, k] < T0:
            ri_k = subW[i, k] if subW[i, k] < wlim_k else wlim_k
            if ri_k < 0.0:
                ri_k = 0.0
        subW[i, k] -= ri_k
        subT[i, k] += (Lm * ri_k) / (subD[i, k] * cpi_k * subZ[i, k])
        subD[i, k] += ri_k / subZ[i, k]
        ri_sum += ri_k

    slushw_i = 0.0
    irrw_i = 0.0
    for k in range(nl):
        slushw_i += subS[i, k]
        irrw_i += subW[i, k]
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
    if i >= subD.shape[0]:
        return
    nl = subD.shape[1]

    # Effective conductivity and volumetric heat capacity per layer.
    for k in range(nl):
        d = subD[i, k]
        kk[i, k] = 0.138 - 1.01e-3 * d + 3.233e-6 * d * d
        c_eff[i, k] = d * (152.2 + 7.122 * subT[i, k])

    # dz1 = (subZ[0] + 0.5*subZ[1])**2
    half1 = subZ[i, 0] + 0.5 * subZ[i, 1]
    dz1[i] = half1 * half1

    # dz2 = 0.5 * (subZ[k+2] + subZ[k+1])**2   (NumPy: 0.5*(a+b)**2, NOT (0.5*(a+b))**2)
    for k in range(nl - 2):
        s = subZ[i, k + 2] + subZ[i, k + 1]
        dz2[i, k] = 0.5 * s * s

    # kk_sz_top = kk[0]*subZ[0] + 0.5*kk[1]*subZ[1]
    kk_sz_top[i] = kk[i, 0] * subZ[i, 0] + 0.5 * kk[i, 1] * subZ[i, 1]

    # kk_sz_interior[j] = kk[j+1]*subZ[j+1] + kk[j+2]*subZ[j+2]
    for k in range(nl - 2):
        kk_sz_interior[i, k] = kk[i, k + 1] * subZ[i, k + 1] + kk[i, k + 2] * subZ[i, k + 2]

    # denom_layer1 = c_eff[1] * (0.5*subZ[0] + 0.5*subZ[1] + 0.25*subZ[2])
    denom_layer1[i] = c_eff[i, 1] * (0.5 * subZ[i, 0] + 0.5 * subZ[i, 1] + 0.25 * subZ[i, 2])

    # denom_interior[k-2] = c_eff[k] * (0.25*subZ[k-1] + 0.5*subZ[k] + 0.25*subZ[k+1]) for k in 2..nl-2
    for k in range(2, nl - 1):
        denom_interior[i, k - 2] = c_eff[i, k] * (0.25 * subZ[i, k - 1] + 0.5 * subZ[i, k] + 0.25 * subZ[i, k + 1])

    # denom_bottom = c_eff[-1] * (0.25*subZ[-2] + 0.75*subZ[-1])
    denom_bottom[i] = c_eff[i, nl - 1] * (0.25 * subZ[i, nl - 2] + 0.75 * subZ[i, nl - 1])

    # dt_stab = 0.5 * min(c_eff[1:]) * min(subZ[1:])**2 / max(kk[1:]) / dayseconds
    min_ceff = math.inf
    min_sz = math.inf
    max_kk = 0.0
    for k in range(1, nl):
        if c_eff[i, k] < min_ceff:
            min_ceff = c_eff[i, k]
        if subZ[i, k] < min_sz:
            min_sz = subZ[i, k]
        if kk[i, k] > max_kk:
            max_kk = kk[i, k]
    dt_stab[i] = 0.5 * min_ceff * (min_sz * min_sz) / max_kk / dayseconds


@cuda.jit
def _heat_boundary_clip_kernel_gpu(subT, Tsurf, subZ, T0):
    """Surface ghost-layer update + melting-point clip. One thread per column.

    Matches the NumPy post-processing at the end of heat_conduction():
        subT[:,0] = Tsurf + (subT[:,1]-Tsurf)/(subZ[:,0]+0.5*subZ[:,1]) * 0.5*subZ[:,0]
        np.clip(subT, None, T0, out=subT)
    """
    i = cuda.grid(1)
    if i >= subT.shape[0]:
        return
    nl = subT.shape[1]

    subT[i, 0] = Tsurf[i] + (subT[i, 1] - Tsurf[i]) / (subZ[i, 0] + 0.5 * subZ[i, 1]) * 0.5 * subZ[i, 0]
    for k in range(nl):
        if subT[i, k] > T0:
            subT[i, k] = T0


# ===========================================================================
# Device-resident state for one LOOP_SNOW step.
#
# SnowDeviceState keeps the subsurface arrays (subT/subD/subZ/subW/subS/
# subTmean/surfH) on the GPU across the three snow sub-steps -- compaction,
# heat conduction, percolation -- so they are uploaded once and downloaded
# once per timestep instead of round-tripping per sub-step.
#
# The host-side physics glue (mode selection, water-input formula, unit
# conventions, output bookkeeping) stays in LOOP_SNOW.py, which is the single
# source of truth shared with the NumPy and Numba paths. This class only owns
# the device buffers and launches the five compute kernels above against them,
# so --dump-reference parity with the NumPy/Numba paths is preserved.
#
# NOTE (lifetime): one instance currently lives for a single LOOP_SNOW step
# (created after melt_sublimation, discarded after percolation), so the
# transfer profile matches the previous fused core -- upload once, download
# once per step. Hoisting the instance out of the time loop so the state
# persists across timesteps -- syncing only subZ/subD/Tsurf at the LOOP_EBM /
# LOOP_mass_balance boundary -- is the follow-up that turns the once-per-step
# transfers into once-per-run.
# ===========================================================================


class SnowDeviceState:
    """GPU-resident subsurface state for LOOP_SNOW.

    Allocates all device buffers once for a given grid shape. Each timestep
    upload() refreshes the resident arrays and this step's inputs,
    compaction() / heat_conduction() / percolation() launch their kernels
    against those buffers, and download() copies every result back into the
    OUT dict in one shot.

    The instance is reused across timesteps (see get_device_state), so the
    device allocations happen once per run instead of once per timestep.
    """

    def __init__(self, gpsum, nl):
        self.shape = (gpsum, nl)
        self.blocks = _blocks(gpsum)

        # Resident state (read/write; refreshed in upload, read back in download).
        self.subT = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.subD = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.subZ = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.subW = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.subS = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.subTmean = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.surfH = cuda.device_array((gpsum,), dtype=np.float64)

        # Per-step inputs (read-only on the device).
        self.logyearsnow = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.yearsnow = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.WS = cuda.device_array((gpsum,), dtype=np.float64)
        self.Tsurf = cuda.device_array((gpsum,), dtype=np.float64)
        self.sumWinit = cuda.device_array((gpsum,), dtype=np.float64)

        # Dens_* diagnostics, refreshed from the host zeros each step.
        self.Dens_destr_metam = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.Dens_overb_pres = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.Dens_drift = cuda.device_array((gpsum, nl), dtype=np.float64)

        # Conductivity / heat capacity filled by the heat-conduction prep kernel
        # and kept resident so download() returns subK/subCeff matching NumPy.
        self.kk = cuda.device_array((gpsum, nl), dtype=np.float64)
        self.c_eff = cuda.device_array((gpsum, nl), dtype=np.float64)

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
        self._subD_old = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._was_snow_ws = cuda.device_array((gpsum, nl), dtype=np.bool_)
        self._psload_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._kk_sz_top = cuda.device_array((gpsum,), dtype=np.float64)
        self._kk_sz_interior = cuda.device_array((gpsum, nl - 2), dtype=np.float64)
        self._dz1 = cuda.device_array((gpsum,), dtype=np.float64)
        self._dz2 = cuda.device_array((gpsum, nl - 2), dtype=np.float64)
        self._denom_layer1 = cuda.device_array((gpsum,), dtype=np.float64)
        self._denom_interior = cuda.device_array((gpsum, nl - 3), dtype=np.float64)
        self._denom_bottom = cuda.device_array((gpsum,), dtype=np.float64)
        self._dt_stab = cuda.device_array((gpsum,), dtype=np.float64)
        self._T_loc_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._kdTdz_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._RP = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._wlim_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._wirr_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._carrot_ws = cuda.device_array((gpsum, nl), dtype=np.float64)
        self._slushspace_ws = cuda.device_array((gpsum, nl), dtype=np.float64)

    def upload(
        self,
        subT,
        subD,
        subZ,
        subW,
        subS,
        subTmean,
        surfH,
        logyearsnow,
        yearsnow,
        WS,
        Tsurf,
        sumWinit,
    ):
        """Refresh the device buffers with this timestep's host state."""
        self.subT.copy_to_device(subT)
        self.subD.copy_to_device(subD)
        self.subZ.copy_to_device(subZ)
        self.subW.copy_to_device(subW)
        self.subS.copy_to_device(subS)
        self.subTmean.copy_to_device(subTmean)
        self.surfH.copy_to_device(surfH)
        self.logyearsnow.copy_to_device(logyearsnow)
        self.yearsnow.copy_to_device(yearsnow)
        self.WS.copy_to_device(WS)
        self.Tsurf.copy_to_device(Tsurf)
        self.sumWinit.copy_to_device(sumWinit)

    def compaction(
        self,
        Dens_destr_metam,
        Dens_overb_pres,
        Dens_drift,
        dt_yearfrac,
        dt_seconds,
        dt,
        C,
        tau_drift,
        mode,
    ):
        """Launch the compaction kernel on the resident arrays."""
        # Upload host-zeroed diagnostics; firn_only leaves them untouched.
        self.Dens_destr_metam.copy_to_device(Dens_destr_metam)
        self.Dens_overb_pres.copy_to_device(Dens_overb_pres)
        self.Dens_drift.copy_to_device(Dens_drift)

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
        d_avail_W = cuda.to_device(avail_W)
        # Snapshot post-heat subW on-device (host path does subW.copy()).
        _percolation_kernel_gpu[self.blocks, _TPB](
            self.subT,
            self.subD,
            self.subW,
            self.subS,
            self.subZ,
            d_avail_W,
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

    def download(self, OUT):
        """Copy the resident state and every output back into OUT (once)."""
        # Resident state: written back in place, preserving array identity.
        self.subT.copy_to_host(OUT["subT"])
        self.subD.copy_to_host(OUT["subD"])
        self.subZ.copy_to_host(OUT["subZ"])
        self.subW.copy_to_host(OUT["subW"])
        self.subS.copy_to_host(OUT["subS"])
        self.subTmean.copy_to_host(OUT["subTmean"])
        self.surfH.copy_to_host(OUT["surfH"])
        self.Dens_destr_metam.copy_to_host(OUT["Dens_destr_metam"])
        self.Dens_overb_pres.copy_to_host(OUT["Dens_overb_pres"])
        self.Dens_drift.copy_to_host(OUT["Dens_drift"])
        self.runoff_irr.copy_to_host(OUT["runoff_irr"])

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

        # subK / subCeff are the post-compaction conductivity and heat capacity
        # (from the prep kernel), matching the NumPy path's pre-heat-solve
        # values. cpi is a diagnostic recomputed from the final subT; as in the
        # per-call GPU path this differs from NumPy only at the last refreezing
        # sub-step and is not consumed downstream.
        self.kk.copy_to_host(OUT["subK"])
        self.c_eff.copy_to_host(OUT["subCeff"])
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


def get_device_state(
    subT,
    subD,
    subZ,
    subW,
    subS,
    subTmean,
    surfH,
    logyearsnow,
    yearsnow,
    WS,
    Tsurf,
    sumWinit,
):
    """Return the cached SnowDeviceState, refreshed with the current host state.

    Allocates on the first call (or if the grid shape changed) and uploads this
    timestep's arrays into the existing device buffers on every call.
    """
    global _device_state
    if _device_state is None or _device_state.shape != subD.shape:
        gpsum, nl = subD.shape
        _device_state = SnowDeviceState(gpsum, nl)
    _device_state.upload(
        subT,
        subD,
        subZ,
        subW,
        subS,
        subTmean,
        surfH,
        logyearsnow,
        yearsnow,
        WS,
        Tsurf,
        sumWinit,
    )
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
