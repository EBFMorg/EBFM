# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Kernels for the LOOP_SNOW module.

Availability: Numba-compiled kernels for the LOOP_SNOW module.
- Compaction
- Heat conduction
- Percolation, refreezing and irreducible water storage

In the future: GPU offloading (e.g. with CuPy or Numba CUDA).

When numba is not installed, the @njit decorator is a no-op (provided by
compute_backend), so the functions remain importable but run as plain Python.
"""

import math

import numpy as np

from .compute_backend import njit, prange


@njit(parallel=True, cache=True)
def _compaction_kernel(
    subD,  # (gpsum, nl) Output array, updated in-place
    subZ,  # (gpsum, nl) Output array, updated in-place
    subT,  # (gpsum, nl)
    subW,  # (gpsum, nl), Output array
    subTmean,  # (gpsum, nl) Output array, updated in-place
    subD_old,  # (gpsum, nl)
    subZ_old,  # (gpsum, nl)
    logyearsnow,  # (gpsum, nl)
    yearsnow,  # (gpsum, nl)
    WS,  # (gpsum,)
    Dens_destr_metam,  # (gpsum, nl)
    Dens_overb_pres,  # (gpsum, nl)
    Dens_drift,  # (gpsum, nl)
    surfH,  # (gpsum,), Output array
    sumWinit,  # (gpsum,)
    runoff_irr,  # (gpsum,), Output array
    dt_yearfrac,
    dt_seconds,
    dt,
    Dice,
    Dfirn,
    Dwater,
    g,
    T0,  # scalar
    rd,  # scalar
    Ec,  # scalar
    Eg,  # scalar
    dayseconds,
    tau_drift,
    compaction_mode,
):
    """Per-column compaction kernel, parallelized over gpsum

    1. Firn compaction
    2. Seasonal snow compaction
      2.1 Densification by destructive metamorphism
      2.2 Densification by Overburden pressure
      2.3 Drifting-snow densification
    3. Layer-thickness and surface height, adjustment and subW clipping to mliqmax.
    """
    gpsum, nl = subD.shape
    for i in prange(gpsum):
        # ------ 1. FIRN COMPACTION ------ #
        for k in range(nl):
            subTmean[i, k] = subTmean[i, k] * (1.0 - dt_yearfrac) + dt_yearfrac * subT[i, k]
            cond_firn_k = (compaction_mode == 0) or (subD[i, k] >= Dfirn)
            if cond_firn_k:
                if subD[i, k] < 550.0:
                    grav_const = 0.07 * max(1.435 - 0.151 * logyearsnow[i, k], 0.25)
                else:
                    grav_const = 0.03 * max(2.366 - 0.293 * logyearsnow[i, k], 0.25)
                temp_factor = math.exp(-Ec / (rd * subT[i, k]) + Eg / (rd * subTmean[i, k]))
                firn_inc = dt_yearfrac * grav_const * yearsnow[i, k] * g * (Dice - subD[i, k]) * temp_factor
                subD[i, k] += firn_inc

        # ------ 2. SEASONAL SNOW COMPACTION ------ #
        if compaction_mode == 1:  # firn+snow
            # ------ 2.1 DENSIFICATION BY DESTRUCTIVE METAMORPHISM ------ #
            # Capture pre-DM snow mask (subD < Dfirn) BEFORE modifying subD.
            # Reuses it for overburden and drifting
            was_snow = np.empty(nl, dtype=np.bool_)
            for k in range(nl):
                was_snow[k] = subD[i, k] < Dfirn

            for k in range(nl):
                if was_snow[k]:
                    cc1 = math.exp(-0.046 * max(subD[i, k] - 175.0, 0.0))
                    cc2 = 1.0 + (1.0 if subW[i, k] != 0.0 else 0.0)
                    temp_exp = math.exp(0.04 * (subT[i, k] - T0))
                    snow_inc = cc1 * cc2 * 2.777e-6 * temp_exp * dt_seconds * subD[i, k]
                    subD[i, k] = min(subD[i, k] + snow_inc, Dice)
                    Dens_destr_metam[i, k] = snow_inc
                else:
                    Dens_destr_metam[i, k] = 0.0

            # ------ 2.2 DENSIFICATION BY OVERBURDEN PRESSURE ------ #
            # Two passes:
            # (1) compute Psload from post-DM densities
            # (2) apply updates
            # Refactoring into a loop introduces slight rounding errors!
            psload = np.empty(nl)
            psload[0] = 0.5 * subD[i, 0] * subZ[i, 0] * g
            for k in range(1, nl):
                xm = subD[i, k - 1] * subZ[i, k - 1] * g
                xk = subD[i, k] * subZ[i, k] * g
                psload[k] = psload[k - 1] + 0.5 * (xm + xk)

            for k in range(nl):
                Dens_overb_pres[i, k] = 0.0
                if was_snow[k]:  # use pre-DM mask to match NumPy cond_snow
                    cc7 = 4.0 * 7.62237e6 / 250.0 * subD[i, k] / (1.0 + 60.0 * subW[i, k] / (Dwater * subZ[i, k]))
                    visc = cc7 * math.exp(0.1 * (T0 - subT[i, k]) + 0.023 * subD[i, k])
                    overb_inc = dt * dayseconds * subD[i, k] * psload[k] / visc
                    subD[i, k] = min(subD[i, k] + overb_inc, Dice)
                    Dens_overb_pres[i, k] = dt * dayseconds * subD[i, k] * psload[k] / visc

            # ------ 2.3 DRIFTING SNOW DENSIFICATION ------ #
            # Refactoring into a loop introduces slight rounding errors!
            # Use updated subD < Dfirn to match cond_drift_total
            z_i_k = 0.0  # z_i[0] = 0
            for k in range(nl):
                d_k = max(subD[i, k], 50.0)
                mo_k = -0.069 + 0.66 * (1.25 - 0.0042 * (d_k - 50.0))
                si_k = -2.868 * math.exp(-0.085 * WS[i]) + 1.0 + mo_k
                gamma_k = max(0.0, si_k * math.exp(-z_i_k / 0.1))
                Dens_drift[i, k] = 0.0
                if si_k > 0.0 and subD[i, k] < Dfirn:  # use updated mask to match NumPy cond_drift_total
                    tau_i_k = tau_drift / gamma_k  # gamma_k > 0 since si_k > 0
                    drift_inc = dt_seconds * max(350.0 - subD[i, k], 0.0) / tau_i_k
                    subD[i, k] = min(subD[i, k] + drift_inc, Dice)
                    Dens_drift[i, k] = drift_inc
                # z_i[k+1] = z_i[k] + subZ[k] * (3.25 - SI[k])
                z_i_k += subZ[i, k] * (3.25 - si_k)

        # ------ 3. UPDATE LAYER THICKNESS & SURFACE HEIGHT AFTER COMPACTION ------
        z_sum = 0.0
        z_sum_old = 0.0
        subW_sum = 0.0
        for k in range(nl):
            if subD[i, k] < Dice:
                subZ[i, k] = subZ_old[i, k] * subD_old[i, k] / subD[i, k]
                exp_f = 0.0143 * math.exp(3.3 * (Dice - subD[i, k]) / Dice)
                denom = 1.0 - exp_f
                mliqmax_k = subD[i, k] * subZ[i, k] * exp_f / denom * 0.05 * min(Dice - subD[i, k], 20.0)
                if subW[i, k] > mliqmax_k:
                    subW[i, k] = mliqmax_k
            else:
                # Ice layer: mliqmax = 0 => clamp subW to zero
                subW[i, k] = 0.0
            z_sum += subZ[i, k]
            z_sum_old += subZ_old[i, k]
            subW_sum += subW[i, k]

        surfH[i] += z_sum - z_sum_old
        runoff_irr[i] = sumWinit[i] - subW_sum


@njit(parallel=True, cache=True)
def _heat_conduction_kernel(
    subT,  # (gpsum, nl) Output array, updated in-place
    Tsurf,  # (gpsum,)
    kk_sz_top,  # (gpsum,)
    kk_sz_mid,  # (gpsum, nl-2)
    dz1,  # (gpsum,)
    dz2,  # (gpsum, nl-2)
    denom_l1,  # (gpsum,)
    denom_interior,  # (gpsum, nl-3)
    denom_bottom,  # (gpsum,)
    dt_stab,  # (gpsum,)
    dt,  # scalar — total time step in days
    dayseconds,  # scalar — seconds per day
    geothermal_flux,  # scalar — W m-2
):
    """Per-column heat-conduction kernel, parallelized over gpsum.

    Each column is independent over gpsum: the CFL sub-stepping while-loop runs entirely
    per grid point with no inter-column communication,

    For each sub-step inside the while-loop:
      1. Compute all inter-layer heat fluxes ``kdTdz`` from the *current*
         column temperatures ``T_loc``.
      2. Update ``T_loc`` in-place using ``kdTdz``.
      3. Repeat until the full time step ``dt`` is covered.
    """
    gpsum, nl = subT.shape
    for i in prange(gpsum):
        # Thread-local working copy and flux array
        T_loc = subT[i, :].copy()
        kdTdz = np.zeros(nl)

        tt_i = 0.0
        while tt_i < dt:
            dt_temp_i = min(dt_stab[i], dt - tt_i)
            if dt_temp_i == 0.0:
                break
            tt_i += dt_temp_i
            C_day_dt = dayseconds * dt_temp_i

            # ---- Step 1: freeze all fluxes from current T_loc ----
            kdTdz[1] = kk_sz_top[i] * (T_loc[1] - Tsurf[i]) / dz1[i]
            for k in range(2, nl):
                kdTdz[k] = kk_sz_mid[i, k - 2] * (T_loc[k] - T_loc[k - 1]) / dz2[i, k - 2]

            # ---- Step 2: update T_loc in-place (kdTdz is now frozen) ----
            T_loc[1] += C_day_dt * (kdTdz[2] - kdTdz[1]) / denom_l1[i]
            for k in range(2, nl - 1):
                T_loc[k] += C_day_dt * (kdTdz[k + 1] - kdTdz[k]) / denom_interior[i, k - 2]
            T_loc[nl - 1] += C_day_dt * (geothermal_flux - kdTdz[nl - 1]) / denom_bottom[i]

        # Write results back to the shared subT array
        for k in range(nl):
            subT[i, k] = T_loc[k]


@njit(parallel=True, cache=True)
def _percolation_kernel(
    subT,  # (gpsum, nl) Output array, updated in-place
    subD,  # (gpsum, nl) Output array, updated in-place
    subW,  # (gpsum, nl) Output array, updated in-place
    subS,  # (gpsum, nl) Output array, rewritten
    subZ,  # (gpsum, nl)
    subW_old,  # (gpsum, nl)
    avail_W,  # (gpsum,)
    RP,  # (gpsum, nl)
    runoff_surface,  # (gpsum,)
    runoff_slush,  # (gpsum,)
    refr_P,  # (gpsum,)
    refr_S,  # (gpsum,)
    refr_I,  # (gpsum,)
    slushw,  # (gpsum,)
    irrw,  # (gpsum,)
    T0,
    Dice,
    Dwater,
    Lm,
    Trunoff,
    perc_depth,
    percolation_mode,  # 0=bucket, 1=normal, 2=linear, 3=uniform
    dt,
):
    """Per-column percolation, slush storage and refreezing kernel, parallelized over gpsum

    1. Compute Wlim and Wirr (refreezing potential and available irreducible water storate)
    2. carrot distribution profile (bucket / normal / linear / uniform)
    3. Refreezing and irreducible-water-storage
    4. Slush storage
    5. Slush refreezing
    6. Irreducible-water refreezing
    """
    gpsum, nl = subT.shape
    sigma2_2 = 2.0 * (perc_depth / 3.0) ** 2
    norm_coeff = 2.0 / (perc_depth / 3.0) / math.sqrt(2.0 * math.pi)
    trunoff_factor = 1.0 / (1.0 + dt / Trunoff)

    for i in prange(gpsum):
        # ------ Refreezing and Irreducible Water Storage Limits ------
        # Compute refreezing potential (`Wlim`) per layer
        # Compute maximum irreducible water storage (`mliqmax`)
        # Compute available irreducible water storage (`Wirr`) per layer
        wlim_loc = np.empty(nl)
        wirr_loc = np.empty(nl)
        for k in range(nl):
            cpi_k = 152.2 + 7.122 * subT[i, k]
            c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
            c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
            wlim_loc[k] = max(min(c1_k, c2_k), 0.0)
            if subD[i, k] < Dice - 1.0:
                factor_k = 3.3 * (Dice - subD[i, k]) / Dice
                exp_f = math.exp(factor_k)
                irr_f = 0.0143 * exp_f / (1.0 - 0.0143 * exp_f)
                mliqmax_k = subD[i, k] * subZ[i, k] * irr_f * 0.05 * min(Dice - subD[i, k], 20.0)
            else:
                mliqmax_k = 0.0
            wirr_loc[k] = mliqmax_k - subW_old[i, k]

        # ------ Compute carrot (water-distribution profile) by percolation mode ------
        carrot_loc = np.zeros(nl)
        if percolation_mode == 0:  # bucket: all water enters surface layer
            carrot_loc[0] = 1.0
        else:
            # Compute zz (midpoint depth of each layer) for mode 1/2/3
            depth = 0.0
            for k in range(nl):
                zz_k = depth + 0.5 * subZ[i, k]
                if percolation_mode == 1:  # normal (Gaussian)
                    carrot_loc[k] = norm_coeff * math.exp(-(zz_k * zz_k) / sigma2_2)
                elif percolation_mode == 2:  # linear
                    v = 2.0 * (perc_depth - zz_k) / (perc_depth * perc_depth)
                    carrot_loc[k] = v if v > 0.0 else 0.0
                else:  # uniform (mode 3): temporarily store zz for the argmin pass below
                    carrot_loc[k] = zz_k
                depth += subZ[i, k]

            if percolation_mode == 3:  # uniform: resolve argmin, then fill layers 0..ind
                min_dist = math.inf
                ind = 0
                for k in range(nl):
                    d = abs(carrot_loc[k] - perc_depth)
                    if d < min_dist:
                        min_dist = d
                        ind = k
                for k in range(nl):
                    carrot_loc[k] = (1.0 / perc_depth) if k <= ind else 0.0

        # Scale by layer thickness, normalize, multiply by avail_W
        s = 0.0
        for k in range(nl):
            carrot_loc[k] *= subZ[i, k]
            s += carrot_loc[k]
        avail_W_i = avail_W[i]
        for k in range(nl):
            carrot_loc[k] = carrot_loc[k] / s * avail_W_i

        #########################################################
        # Percolation loop: top-to-bottom refreezing + irreducible storage
        # avail_W_loc carries unabsorbed water forward across layers
        #########################################################
        avail_W_loc = 0.0
        rp_sum = 0.0
        for n in range(nl):
            avail_W_loc += carrot_loc[n]
            rp_n = min(avail_W_loc, wlim_loc[n])
            RP[i, n] = rp_n
            excess = avail_W_loc - wlim_loc[n]
            if excess < 0.0:
                excess = 0.0
            new_subW_n = subW_old[i, n] + min(excess, wirr_loc[n])
            subW[i, n] = new_subW_n
            avail_W_loc -= rp_n + (new_subW_n - subW_old[i, n])
            # Temperature and density update after percolating-water refreezing
            cpi_n = 152.2 + 7.122 * subT[i, n]
            subT[i, n] += Lm * rp_n / (subD[i, n] * cpi_n * subZ[i, n])
            subD[i, n] += rp_n / subZ[i, n]
            rp_sum += rp_n

        avail_W[i] = avail_W_loc  # write leftover back (becomes avail_W for slush section)

        #########################################################
        # Slush water storage
        #########################################################
        slushspace = np.empty(nl)
        total_slushspace = 0.0
        for k in range(nl):
            ss_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dwater - subW[i, k]
            if ss_k < 0.0:
                ss_k = 0.0
            slushspace[k] = ss_k
            total_slushspace += ss_k

        # Old slush + leftover water form the available slush input
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

        # Bottom-up fill of slush pore space
        for n in range(nl - 1, -1, -1):
            fill = avail_S if avail_S < slushspace[n] else slushspace[n]
            subS[i, n] = fill
            avail_S -= fill

        #####################################
        # Refreezing of slush water
        #####################################
        rs_sum = 0.0
        for k in range(nl):
            cpi_k = 152.2 + 7.122 * subT[i, k]
            c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
            c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
            wlim_k = min(c1_k, c2_k)
            rs_k = 0.0
            if subS[i, k] > 0.0 and subT[i, k] < T0:
                rs_k = subS[i, k] if subS[i, k] < wlim_k else wlim_k
                if rs_k < 0.0:
                    rs_k = 0.0
            subS[i, k] -= rs_k
            subT[i, k] += (Lm * rs_k) / (subD[i, k] * cpi_k * subZ[i, k])
            subD[i, k] += rs_k / subZ[i, k]
            rs_sum += rs_k

        #########################################################
        # Irreducible water refreezing
        #########################################################
        ri_sum = 0.0
        for k in range(nl):
            cpi_k = 152.2 + 7.122 * subT[i, k]
            c1_k = cpi_k * subD[i, k] * subZ[i, k] * (T0 - subT[i, k]) / Lm
            c2_k = subZ[i, k] * (1.0 - subD[i, k] / Dice) * Dice
            wlim_k = min(c1_k, c2_k)
            ri_k = 0.0
            if subW[i, k] > 0.0 and subT[i, k] < T0:
                ri_k = subW[i, k] if subW[i, k] < wlim_k else wlim_k
                if ri_k < 0.0:
                    ri_k = 0.0
            subW[i, k] -= ri_k
            subT[i, k] += (Lm * ri_k) / (subD[i, k] * cpi_k * subZ[i, k])
            subD[i, k] += ri_k / subZ[i, k]
            ri_sum += ri_k

        #########################################################
        # Scalar outputs per column
        #########################################################
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
