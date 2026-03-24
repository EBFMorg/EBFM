# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import math

from .constants import SECONDS_PER_HOUR

# line_profiler support: `profile` is injected as a builtin by kernprof.
# When running normally, fall back to a no-op so the decorator stays in place.
try:
    profile  # noqa: F821
except NameError:
    profile = lambda f: f  # noqa: E731

# ---------------------------------------------------------------------------
# Optional Numba support
# Install via:  pip install "ebfm[performance]"
# Without numba the module falls back to the NumPy path automatically.
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

        # Handle both @njit and @njit(parallel=True, ...) call styles
        return _wrap if kwargs or (args and not callable(args[0])) else args[0]

    prange = range  # type: ignore[assignment]  # noqa: F811

# Dispatch flag — False by default (opt-in via --with-numba).
# Set to True by main.py when --with-numba is passed and numba is properly available.
_USE_NUMBA = False

# Optional per-function diagnostic dumps.
# Set to a directory path string to save subT/subD/subZ/subW after every inner
# function call, e.g.:
#   import ebfm.core.LOOP_SNOW as LS; LS._DIAG_DUMP = "/tmp/numpy"
# Then run again with _USE_NUMBA=True and LS._DIAG_DUMP = "/tmp/numba".
# Compare pairs with: numpy.testing.assert_allclose or compare_snapshots.py.
_DIAG_DUMP = None  # type: str | None
_DIAG_MAX_STEPS = 1  # only dump this many timesteps (0 = unlimited)


def _diag_dump(label: str, OUT: dict) -> None:
    """Save tracked arrays to _DIAG_DUMP/<label>.npz when _DIAG_DUMP is set.

    Only writes for timesteps 0 .. _DIAG_MAX_STEPS-1.
    Set _DIAG_MAX_STEPS = 0 to dump every timestep.
    """
    if _DIAG_DUMP is None:
        return
    if _DIAG_MAX_STEPS > 0:
        return
    import os

    os.makedirs(_DIAG_DUMP, exist_ok=True)
    keys = [
        "smb",
        "smb_cumulative",
        "Tsurf",
        "subT",
        "subD",
        "subZ",
        "subW",
        "subS",
        "surfH",
        "subTmean",
        "runoff_irr",
        "Dens_destr_metam",
        "Dens_overb_pres",
        "Dens_drift",
        # reboot / restart state
        "snowmass",
        "ys",
        "timelastsnow_netCDF",
        "alb_snow",
    ]
    arrays = {k: OUT[k].copy() for k in keys if k in OUT}
    path = os.path.join(_DIAG_DUMP, f"step_{label}.npz")
    np.savez_compressed(path, **arrays)
    print(f"[DIAG] saved {path}")


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


def main(C, OUT, IN, dt, grid, phys):
    """
    Implementation of the multi-layer snow and firn model

    Parameters:
        C (dict): Model constants and parameters.
        OUT (dict): Output variables to store results.
        IN (dict): Input data for the model.
        dt: Model time-step.
        phys (dict): Model physics settings.

    Returns:
        dict: Updated OUT dictionary.
    """

    def snowfall_and_deposition():
        """
        Calculate snowfall and deposition and shift vertical grid accordingly
        """

        max_subZ = grid["max_subZ"]
        gpsum = grid["gpsum"]
        nl = grid["nl"]

        # Fresh snow density calculations
        if phys["snow_compaction"] == "firn+snow":
            OUT["Dfreshsnow_T"] = np.zeros_like(IN["T"])

            temp_above = IN["T"] > C["T0"] + 2
            temp_between = (IN["T"] <= C["T0"] + 2) & (IN["T"] > C["T0"] - 15)
            temp_below = IN["T"] <= C["T0"] - 15

            OUT["Dfreshsnow_T"][temp_above] = 50 + 1.7 * 17 ** (3 / 2)
            OUT["Dfreshsnow_T"][temp_between] = 50 + 1.7 * (IN["T"][temp_between] - C["T0"] + 15) ** (3 / 2)
            OUT["Dfreshsnow_T"][temp_below] = (
                -3.8328 * (IN["T"][temp_below] - C["T0"]) - 0.0333 * (IN["T"][temp_below] - C["T0"]) ** 2
            )

            OUT["Dfreshsnow_W"] = 266.86 * (0.5 * (1 + np.tanh(IN["WS"] / 5))) ** 8.8
            OUT["Dfreshsnow"] = OUT["Dfreshsnow_T"] + OUT["Dfreshsnow_W"]
        else:
            OUT["Dfreshsnow"] = np.full_like(IN["snow"], C["Dfreshsnow"])

        # Update layer depths and properties
        shift_snowfall = IN["snow"] * C["Dwater"] / OUT["Dfreshsnow"]
        shift_riming = OUT["moist_deposition"] * C["Dwater"] / OUT["Dfreshsnow"]
        shift_tot = shift_snowfall + shift_riming
        OUT["surfH"] += shift_tot

        OUT["runoff_irr_deep"] = np.zeros(gpsum)

        # Main processing loop: Run until all shifts are handled
        while np.any(shift_tot > 0):
            shift = np.minimum(shift_tot, max_subZ)
            shift_tot -= shift

            # Use references instead of unnecessary .copy()
            subT_old = OUT["subT"].copy()
            subD_old = OUT["subD"].copy()
            subW_old = OUT["subW"].copy()
            subZ_old = OUT["subZ"].copy()

            # Precompute conditions for better performance
            is_noshift = subZ_old[:, 0] + shift <= max_subZ
            is_shift = ~is_noshift

            # Handle no-shift updates (vectorized)
            OUT["subZ"][is_noshift, 0] += shift[is_noshift]
            OUT["subT"][is_noshift, 0] = (
                subT_old[is_noshift, 0] * subZ_old[is_noshift, 0] / OUT["subZ"][is_noshift, 0]
                + OUT["Tsurf"][is_noshift] * shift[is_noshift] / OUT["subZ"][is_noshift, 0]
            )
            OUT["subD"][is_noshift, 0] = (
                subD_old[is_noshift, 0] * subZ_old[is_noshift, 0] / OUT["subZ"][is_noshift, 0]
                + OUT["Dfreshsnow"][is_noshift] * shift[is_noshift] / OUT["subZ"][is_noshift, 0]
            )

            # Handle shifting updates (vectorized)
            if np.any(is_shift):
                OUT["subZ"][is_shift, 2 : nl - 1] = subZ_old[is_shift, 1 : nl - 2]
                OUT["subT"][is_shift, 2 : nl - 1] = subT_old[is_shift, 1 : nl - 2]
                OUT["subD"][is_shift, 2 : nl - 1] = subD_old[is_shift, 1 : nl - 2]
                OUT["subW"][is_shift, 2 : nl - 1] = subW_old[is_shift, 1 : nl - 2]

                OUT["subZ"][is_shift, 1] = max_subZ
                OUT["subZ"][is_shift, 0] = (subZ_old[is_shift, 0] + shift[is_shift]) - max_subZ
                OUT["subT"][is_shift, 1] = (
                    subT_old[is_shift, 0] * subZ_old[is_shift, 0] / OUT["subZ"][is_shift, 1]
                    + OUT["Tsurf"][is_shift]
                    * (OUT["subZ"][is_shift, 1] - subZ_old[is_shift, 0])
                    / OUT["subZ"][is_shift, 1]
                )
                OUT["subT"][is_shift, 0] = OUT["Tsurf"][is_shift]
                OUT["subD"][is_shift, 1] = (
                    subD_old[is_shift, 0] * subZ_old[is_shift, 0] / OUT["subZ"][is_shift, 1]
                    + OUT["Dfreshsnow"][is_shift]
                    * (OUT["subZ"][is_shift, 1] - subZ_old[is_shift, 0])
                    / OUT["subZ"][is_shift, 1]
                )
                OUT["subD"][is_shift, 0] = OUT["Dfreshsnow"][is_shift]
                OUT["subW"][is_shift, 1] = subW_old[is_shift, 0]
                OUT["subW"][is_shift, 0] = 0.0

            # Update runoff for shifted layers
            OUT["runoff_irr_deep"][is_shift] += subW_old[is_shift, nl - 1]

        return True

    def melt_sublimation():
        """
        Calculate melt and sublimation and shift vertical grid accordingly
        """

        # Initialize variables
        OUT["sumWinit"] = np.sum(OUT["subW"], axis=1)
        mass_removed = (OUT["melt"] + OUT["moist_sublimation"]) * 1e3
        mass_layer = OUT["subD"] * OUT["subZ"]

        shift_tot = np.zeros_like(mass_removed)
        n = 0

        # While there is still mass to remove
        while np.any(mass_removed > 0):
            n += 1
            cond1 = mass_removed > mass_layer[:, n - 1]
            cond2 = (~cond1) & (mass_removed > 0)

            # Update for layers fully removed
            mass_removed[cond1] -= OUT["subD"][cond1, n - 1] * OUT["subZ"][cond1, n - 1]
            shift_tot[cond1] -= OUT["subZ"][cond1, n - 1]

            # Update for layers partially removed
            shift_tot[cond2] -= (mass_removed[cond2] / mass_layer[cond2, n - 1]) * OUT["subZ"][cond2, n - 1]
            mass_removed[cond2] = 0.0

        # While there are shifts required
        while np.any(shift_tot < 0):
            shift = np.maximum(shift_tot, -OUT["subZ"][:, 1])
            shift_tot -= shift

            OUT["surfH"] += shift

            # Save old values for updates
            subT_old = OUT["subT"].copy()
            subD_old = OUT["subD"].copy()
            subW_old = OUT["subW"].copy()
            subZ_old = OUT["subZ"].copy()

            # Find no-shift and shift indices
            i_noshift = np.where(subZ_old[:, 0] + shift > 1e-17)
            i_shift = np.where(subZ_old[:, 0] + shift <= 1e-17)

            # Handle the no-shift case
            OUT["subZ"][i_noshift, 0] = subZ_old[i_noshift, 0] + shift[i_noshift]
            OUT["subT"][i_noshift, 0] = subT_old[i_noshift, 0]
            OUT["subD"][i_noshift, 0] = subD_old[i_noshift, 0]
            temp = OUT["subZ"][i_noshift, 0] / subZ_old[i_noshift, 0]
            OUT["subW"][i_noshift, 0] = subW_old[i_noshift, 0] * temp

            # Handle the shift case
            nl = grid["nl"]
            OUT["subZ"][i_shift, 1 : nl - 2] = subZ_old[i_shift, 2 : nl - 1]
            OUT["subT"][i_shift, 1 : nl - 2] = subT_old[i_shift, 2 : nl - 1]
            OUT["subD"][i_shift, 1 : nl - 2] = subD_old[i_shift, 2 : nl - 1]
            OUT["subW"][i_shift, 1 : nl - 2] = subW_old[i_shift, 2 : nl - 1]

            OUT["subZ"][i_shift, 0] = subZ_old[i_shift, 0] + subZ_old[i_shift, 1] + shift[i_shift]
            OUT["subT"][i_shift, 0] = subT_old[i_shift, 1]
            OUT["subD"][i_shift, 0] = subD_old[i_shift, 1]
            temp = OUT["subZ"][i_shift, 0] / subZ_old[i_shift, 1]
            OUT["subW"][i_shift, 0] = subW_old[i_shift, 1] * temp
            OUT["subT"][i_shift, nl - 1] = subT_old[i_shift, nl - 1]
            OUT["subW"][i_shift, nl - 1] = 0.0

            # Update the deepest layer properties
            for idx in i_shift:
                if grid["doubledepth"] == 1:
                    OUT["subZ"][idx, nl - 1] = 2.0 ** len(grid["split"]) * grid["max_subZ"]
                else:
                    OUT["subZ"][idx, nl - 1] = grid["max_subZ"]
                OUT["subD"][idx, nl - 1] = subD_old[idx, nl - 1]

        return True

    def compaction():
        """
        Calculate snow and firn compaction and update density and layer thickness
        """

        gpsum, nl = grid["gpsum"], grid["nl"]
        Dice, Dfirn = C["Dice"], C["Dfirn"]

        subD_old = OUT["subD"].copy()
        subZ_old = OUT["subZ"].copy()
        mliqmax = np.zeros((gpsum, nl))

        dt_yearfrac = dt / C["yeardays"]
        dt_seconds = dt * C["dayseconds"]

        # ------ FIRN COMPACTION ------ #
        if phys["snow_compaction"] in ["firn_only", "firn+snow"]:
            # Pre-compute the logical condition based on the snow compaction type
            if phys["snow_compaction"] == "firn_only":
                cond_firn = np.ones_like(OUT["subD"], dtype=bool)  # All values are True in 2D
            else:  # 'firn+snow'
                cond_firn = OUT["subD"] >= Dfirn  # Results in the same 2D shape as OUT['subD']

            # Update annual running average subsurface temperature
            OUT["subTmean"] *= 1 - dt_yearfrac
            OUT["subTmean"] += dt_yearfrac * OUT["subT"]

            # Set gravitational constants
            subD_cond = np.where(cond_firn, OUT["subD"], 0)  # Using a masked version of subD
            logyearsnow_cond = np.where(cond_firn, IN["logyearsnow"], 0)
            grav_const = np.zeros_like(OUT["subD"])  # Allocation happens here.
            low_density_mask = cond_firn & (subD_cond < 550)
            high_density_mask = cond_firn & (subD_cond >= 550)
            grav_const[low_density_mask] = 0.07 * np.maximum(1.435 - 0.151 * logyearsnow_cond[low_density_mask], 0.25)
            grav_const[high_density_mask] = 0.03 * np.maximum(2.366 - 0.293 * logyearsnow_cond[high_density_mask], 0.25)

            # Update firn densities
            temp_factor = np.exp(-C["Ec"] / (C["rd"] * OUT["subT"]) + C["Eg"] / (C["rd"] * OUT["subTmean"]))
            firn_increment = dt_yearfrac * grav_const * IN["yearsnow"] * C["g"] * (Dice - OUT["subD"]) * temp_factor
            OUT["subD"][cond_firn] += firn_increment[cond_firn]
        else:
            raise ValueError("phys.snow_compaction not set correctly!")

        # ------ SEASONAL SNOW COMPACTION ------ #
        if phys["snow_compaction"] == "firn+snow":
            # ------ DENSIFICATION BY DESTRUCTIVE METAMORPHISM ------ #
            # Precompute condition for snow compaction
            cond_snow = OUT["subD"] < Dfirn

            # Constants for snow compaction
            CC3, CC4 = 2.777e-6, 0.04

            # Precompute CC1, CC2, and temp_exp
            CC1 = np.exp(-0.046 * np.clip(OUT["subD"] - 175, 0, None))
            CC2 = 1 + (OUT["subW"] != 0)
            temp_exp = np.exp(CC4 * (OUT["subT"] - C["T0"]))  #

            # Snow densification increment
            snow_increment = CC1 * CC2 * CC3 * temp_exp * dt_seconds * OUT["subD"]

            # Apply snow increment only to relevant layers
            OUT["subD"][cond_snow] += snow_increment[cond_snow]
            OUT["subD"][cond_snow] = np.minimum(OUT["subD"][cond_snow], Dice)

            # Store densification by destructive metamorphism
            OUT["Dens_destr_metam"] = np.zeros_like(OUT["subD"])
            OUT["Dens_destr_metam"][cond_snow] = snow_increment[cond_snow]

            # ------ DENSIFICATION BY OVERBURDEN PRESSURE ------ #
            CC5, CC6 = 0.1, 0.023
            CC7 = 4.0 * 7.62237e6 / 250.0 * OUT["subD"] * 1 / (1 + 60 * OUT["subW"] * 1 / (C["Dwater"] * OUT["subZ"]))

            # Compute load pressure (Psload)
            OUT_subD_Z = OUT["subD"] * OUT["subZ"] * C["g"]
            Psload = np.cumsum(OUT_subD_Z, axis=1) - 0.5 * OUT_subD_Z
            Psload[~cond_snow] = 0

            # Compute viscosity (Visc)
            temperature_diff = C["T0"] - OUT["subT"]
            Visc = CC7 * np.exp(CC5 * temperature_diff + CC6 * OUT["subD"])
            Visc[~cond_snow] = 0

            # Update densities
            OUT["subD"][cond_snow] += (
                dt * C["dayseconds"] * OUT["subD"][cond_snow] * Psload[cond_snow] / Visc[cond_snow]
            )
            OUT["subD"][cond_snow] = np.minimum(OUT["subD"][cond_snow], C["Dice"])

            # Store densification by overburden pressure
            OUT["Dens_overb_pres"] = np.zeros_like(OUT["subD"])
            OUT["Dens_overb_pres"][cond_snow] = (
                dt * C["dayseconds"] * OUT["subD"][cond_snow] * Psload[cond_snow] / Visc[cond_snow]
            )

            # ------ DRIFTING SNOW DENSIFICATION ------ #
            MO = -0.069 + 0.66 * (1.25 - 0.0042 * (np.maximum(OUT["subD"], 50) - 50))
            SI = -2.868 * np.exp(-0.085 * np.tile(IN["WS"], (nl, 1)).T) + 1 + MO
            cond_drift = SI > 0

            z_i = np.zeros_like(OUT["subZ"])
            if nl > 1:
                z_i[:, 1:] = np.cumsum(OUT["subZ"][:, :-1] * (3.25 - SI[:, :-1]), axis=1)
            gamma_drift = np.maximum(0, SI * np.exp(-z_i / 0.1))
            tau = 48 * 2 * SECONDS_PER_HOUR
            np.seterr(divide="ignore")
            tau_i = tau / gamma_drift

            # Update densities
            drift_increment = dt_seconds * np.maximum(350 - OUT["subD"], 0) / tau_i
            cond_drift_total = cond_drift & (OUT["subD"] < Dfirn)
            OUT["subD"][cond_drift_total] += drift_increment[cond_drift_total]
            OUT["subD"][cond_drift_total] = np.minimum(OUT["subD"][cond_drift_total], Dice)

            # Store densification by wind shearing
            OUT["Dens_drift"] = np.zeros_like(OUT["subD"])
            OUT["Dens_drift"][cond_drift_total] = drift_increment[cond_drift_total]

        # ------ UPDATE LAYER THICKNESS & SURFACE HEIGHT AFTER COMPACTION ------ #
        cond_layers = OUT["subD"] < Dice

        # Update layer thickness
        OUT["subZ"][cond_layers] = subZ_old[cond_layers] * subD_old[cond_layers] / OUT["subD"][cond_layers]

        # Update irreducible water storage
        exp_factor = 0.0143 * np.exp(3.3 * (Dice - OUT["subD"][cond_layers]) / Dice)
        mliqmax[cond_layers] = (
            OUT["subD"][cond_layers]
            * OUT["subZ"][cond_layers]
            * exp_factor
            / (1 - exp_factor)
            * 0.05
            * np.minimum(Dice - OUT["subD"][cond_layers], 20)
        )
        OUT["subW"] = np.minimum(mliqmax, OUT["subW"])

        # Update surface height and runoff
        shift = np.sum(OUT["subZ"], axis=1) - np.sum(subZ_old, axis=1)
        OUT["surfH"] += shift
        OUT["runoff_irr"] = OUT["sumWinit"] - np.sum(OUT["subW"], axis=1)

        return True

    @profile
    def heat_conduction():
        """
        Calculate heat diffusion and update temperatures
        """
        dz1 = (OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1]) ** 2
        dz2 = 0.5 * (OUT["subZ"][:, 2:] + OUT["subZ"][:, 1:-1]) ** 2
        kk = 0.138 - 1.01e-3 * OUT["subD"] + 3.233e-6 * OUT["subD"] ** 2  # Effective conductivity
        c_eff = OUT["subD"] * (152.2 + 7.122 * OUT["subT"])  # Effective heat capacity

        # Stability time step (CFL condition)
        z_temp = OUT["subZ"][:, 1:]
        c_eff_temp = c_eff[:, 1:]
        kk_temp = kk[:, 1:]
        dt_stab = (
            0.5 * np.min(c_eff_temp, axis=1) * np.min(z_temp, axis=1) ** 2 / np.max(kk_temp, axis=1) / C["dayseconds"]
        )

        # ------ Heat Conduction Loop ------
        tt = np.zeros(grid["gpsum"])
        cond_dt_temp = np.zeros_like(tt, dtype=bool)
        kdTdz = np.zeros_like(OUT["subT"])

        while np.any(tt < dt):
            subT_old = OUT["subT"].copy()
            dt_temp = np.minimum(dt_stab, dt - tt)
            tt += dt_temp
            cond_dt = dt_temp > 0
            cond_dt_temp[:] = cond_dt  # Reuse mask to reduce allocations

            # Calculate vertical heat fluxes
            kdTdz[cond_dt, 1] = (
                (kk[cond_dt, 0] * OUT["subZ"][cond_dt, 0] + 0.5 * kk[cond_dt, 1] * OUT["subZ"][cond_dt, 1])
                * (subT_old[cond_dt, 1] - OUT["Tsurf"][cond_dt])
                / dz1[cond_dt]
            )

            kdTdz[cond_dt, 2:] = (
                (kk[cond_dt, 1:-1] * OUT["subZ"][cond_dt, 1:-1] + kk[cond_dt, 2:] * OUT["subZ"][cond_dt, 2:])
                * (subT_old[cond_dt, 2:] - subT_old[cond_dt, 1:-1])
                / dz2[cond_dt]
            )

            # Update layer-wise temperatures
            C_day_dt = C["dayseconds"] * dt_temp[cond_dt]
            OUT["subT"][cond_dt, 1] = subT_old[cond_dt, 1] + C_day_dt * (kdTdz[cond_dt, 2] - kdTdz[cond_dt, 1]) / (
                c_eff[cond_dt, 1]
                * (0.5 * OUT["subZ"][cond_dt, 0] + 0.5 * OUT["subZ"][cond_dt, 1] + 0.25 * OUT["subZ"][cond_dt, 2])
            )

            OUT["subT"][cond_dt, 2:-1] = subT_old[cond_dt, 2:-1] + C_day_dt[:, np.newaxis] * (
                kdTdz[cond_dt, 3:] - kdTdz[cond_dt, 2:-1]
            ) / (
                c_eff[cond_dt, 2:-1]
                * (
                    0.25 * OUT["subZ"][cond_dt, 1:-2]
                    + 0.5 * OUT["subZ"][cond_dt, 2:-1]
                    + 0.25 * OUT["subZ"][cond_dt, 3:]
                )
            )

            OUT["subT"][cond_dt, -1] = subT_old[cond_dt, -1] + C_day_dt * (
                C["geothermal_flux"] - kdTdz[cond_dt, -1]
            ) / (c_eff[cond_dt, -1] * (0.25 * OUT["subZ"][cond_dt, -2] + 0.75 * OUT["subZ"][cond_dt, -1]))

        OUT["subT"][:, 0] = (
            OUT["Tsurf"]
            + (OUT["subT"][:, 1] - OUT["Tsurf"])
            / (OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1])
            * 0.5
            * OUT["subZ"][:, 0]
        )

        # Ensure temperatures do not exceed melting point
        np.clip(OUT["subT"], None, C["T0"], out=OUT["subT"])

        # Store effective conductivity and specific heat capacity
        OUT["subCeff"] = c_eff
        OUT["subK"] = kk

        return True

    @profile
    def percolation_refreezing_and_storage():
        #########################################################
        # Percolation, refreezing and irreducible water storage
        #########################################################
        subW_old = OUT["subW"].copy()  # Store the old water content
        gpsum, nl = OUT["subT"].shape

        if _USE_NUMBA:
            # Numba parallel path:
            _p_mode = {"bucket": 0, "normal": 1, "linear": 2, "uniform": 3}.get(phys["percolation"], -1)
            if _p_mode < 0:
                raise ValueError(f"_percolation_kernel: unknown percolation={phys['percolation']!r}")
            _avail_W = np.maximum(
                OUT["melt"] * 1e3 + IN["rain"] * 1e3 + (OUT["moist_condensation"] - OUT["moist_evaporation"]) * 1e3,
                0.0,
            )
            # Ensure persistent arrays are allocated
            _rp_shape = OUT["subZ"].shape
            if "_perc_RP" not in OUT or OUT["_perc_RP"].shape != _rp_shape:
                OUT["_perc_RP"] = np.zeros(_rp_shape)
            if "subS" not in OUT or OUT["subS"].shape != (gpsum, nl):
                OUT["subS"] = np.zeros((gpsum, nl))
            _runoff_surface = np.empty(gpsum)
            _runoff_slush = np.empty(gpsum)
            _refr_P = np.empty(gpsum)
            _refr_S = np.empty(gpsum)
            _refr_I = np.empty(gpsum)
            _slushw = np.empty(gpsum)
            _irrw = np.empty(gpsum)

            _percolation_kernel(
                OUT["subT"],
                OUT["subD"],
                OUT["subW"],
                OUT["subS"],
                OUT["subZ"],
                subW_old,
                _avail_W,
                OUT["_perc_RP"],
                _runoff_surface,
                _runoff_slush,
                _refr_P,
                _refr_S,
                _refr_I,
                _slushw,
                _irrw,
                C["T0"],
                C["Dice"],
                C["Dwater"],
                C["Lm"],
                C["Trunoff"],
                C["perc_depth"],
                _p_mode,
                dt,
            )
            OUT["runoff_surface"] = _runoff_surface
            OUT["runoff_slush"] = _runoff_slush
            OUT["refr_P"] = _refr_P
            OUT["refr_S"] = _refr_S
            OUT["refr_I"] = _refr_I
            OUT["refr"] = _refr_P + _refr_S + _refr_I
            OUT["slushw"] = _slushw
            OUT["irrw"] = _irrw
            OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]
            return True

        # Original NumPy path:
        # ------ Water Input ------
        avail_W = (
            OUT["melt"] * 1e3  # Meltwater
            + IN["rain"] * 1e3  # Rainfall
            + (OUT["moist_condensation"] - OUT["moist_evaporation"]) * 1e3  # Condensation or evaporation
        )
        avail_W = np.maximum(avail_W, 0)  # Ensure no negative water availability

        # ------ Refreezing and Irreducible Water Storage Limits ------
        OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]  # Specific heat capacity
        c1 = OUT["cpi"] * OUT["subD"] * OUT["subZ"] * (C["T0"] - OUT["subT"]) / C["Lm"]
        c2 = OUT["subZ"] * (1 - OUT["subD"] / C["Dice"]) * C["Dice"]

        # Compute refreezing potential (`Wlim`) per layer
        # np.minimum(c1, c2) is equivalent to np.where(c1 >= c2, c2, c1)
        Wlim = np.maximum(np.minimum(c1, c2), 0)

        # Maximum irreducible water storage (`mliqmax`)
        mliqmax = np.zeros_like(OUT["subD"])
        noice = OUT["subD"] < (C["Dice"] - 1)
        factor = 3.3 * (C["Dice"] - OUT["subD"][noice]) / C["Dice"]
        exp_factor = np.exp(factor)
        irr_factor = 0.0143 * exp_factor / (1 - 0.0143 * exp_factor)
        mliqmax[noice] = (
            OUT["subD"][noice] * OUT["subZ"][noice] * irr_factor * 0.05 * np.minimum(C["Dice"] - OUT["subD"][noice], 20)
        )

        # Available irreducible water storage
        Wirr = mliqmax - subW_old

        # ------ Water Percolation ------
        z0 = C["perc_depth"]
        zz = np.cumsum(OUT["subZ"], axis=1) - 0.5 * OUT["subZ"]
        carrot = np.zeros_like(OUT["subZ"])

        if phys["percolation"] == "bucket":
            carrot[:, 0] = 1  # All water is added at the surface layer
        elif phys["percolation"] == "normal":
            carrot = 2 * np.exp(-(zz**2) / (2 * (z0 / 3) ** 2)) / (z0 / 3) / np.sqrt(2 * np.pi)
        elif phys["percolation"] == "linear":
            carrot = 2 * (z0 - zz) / z0**2
            carrot = np.maximum(carrot, 0)
        elif phys["percolation"] == "uniform":
            ind = np.argmin(np.abs(zz - z0), axis=1)
            carrot[np.arange(carrot.shape[0]), : ind + 1] = 1 / z0
        else:
            raise ValueError("`phys['percolation']` is not set correctly!")

        carrot *= OUT["subZ"]  # Scale by layer thickness
        carrot /= np.sum(carrot, axis=1)[:, np.newaxis]  # Normalize per layer
        carrot *= avail_W[:, np.newaxis]  # Distribute water input among layers

        # ------ Refreezing and Irreducible Water Storage Iteration ------
        # RP reused across calls: same check-and-fill pattern as Dens_* in compaction
        _rp_shape = OUT["subZ"].shape
        if "_perc_RP" not in OUT or OUT["_perc_RP"].shape != _rp_shape:
            OUT["_perc_RP"] = np.zeros(_rp_shape)
        else:
            OUT["_perc_RP"].fill(0.0)
        RP = OUT["_perc_RP"]

        avail_W_loc = np.zeros(grid["gpsum"])  # Available water per layer

        for n in range(nl):
            # Compute available water per layer
            avail_W_loc += carrot[:, n]

            # RP = min(available water, refreezing capacity), correct for both cond1 and ~cond1
            # No branching: when avail <= Wlim the full amount refreezes; when avail > Wlim
            # only Wlim refreezes and the excess goes into irreducible storage.
            np.minimum(avail_W_loc, Wlim[:, n], out=RP[:, n])
            # Excess water after refreezing, clamped to zero when there is none
            excess = np.maximum(avail_W_loc - Wlim[:, n], 0.0)
            OUT["subW"][:, n] = subW_old[:, n] + np.minimum(excess, Wirr[:, n])

            # Deduct water consumed (refrozen + stored) from the running total
            avail_W_loc -= RP[:, n] + (OUT["subW"][:, n] - subW_old[:, n])

            # Update temperature and density after refreezing
            refreeze_heat = C["Lm"] * RP[:, n]
            OUT["subT"][:, n] += refreeze_heat / (OUT["subD"][:, n] * OUT["cpi"][:, n] * OUT["subZ"][:, n])
            OUT["subD"][:, n] += RP[:, n] / OUT["subZ"][:, n]

        # Update leftover water
        avail_W = avail_W_loc

        #########################################################
        # Slush water storage
        #########################################################

        # Calculate available pore space for storing slush water
        slushspace = np.maximum(
            OUT["subZ"] * (1 - OUT["subD"] / C["Dice"]) * C["Dwater"] - OUT["subW"], 0.0
        )  # shape: [grid.gpsum, nl]

        # Total available slush space across all layers
        total_slushspace = np.sum(slushspace, axis=1)  # shape: [grid.gpsum]

        # Calculate surface runoff (excess water at the surface)
        avail_W += np.sum(OUT["subS"], axis=1)  # Update available water with slush from all layers
        OUT["runoff_surface"] = np.maximum(avail_W - total_slushspace, 0.0)  # Excess water at the surface

        # Update slush water content after new water input and runoff
        avail_S = np.minimum(avail_W, total_slushspace)  # Available slush water for storage

        # Calculate slush water runoff and reduce avail_S accordingly
        OUT["runoff_slush"] = avail_S - 1.0 / (1.0 + dt / C["Trunoff"]) * avail_S
        avail_S = 1.0 / (1.0 + dt / C["Trunoff"]) * avail_S
        avail_S[avail_S < 1e-25] = 0.0  # Set near-zero available slush to zero

        # Initialize slush water in all layers (reusing arrays)
        if "subS" not in OUT or OUT["subS"].shape != (gpsum, nl):
            OUT["subS"] = np.zeros((gpsum, nl))
        else:
            OUT["subS"].fill(0.0)

        # Bottom-up filling of pore space with slush water
        # Each layer takes min(available, pore space), no branching needed
        for n in range(nl - 1, -1, -1):  # Loop from bottom (nl) to top (1)
            np.minimum(avail_S, slushspace[:, n], out=OUT["subS"][:, n])
            avail_S -= OUT["subS"][:, n]

        #####################################
        # Refreezing of slush water
        #####################################
        # Determine whether cold content or density limits the amount of refreezing
        OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]
        c1 = OUT["cpi"] * OUT["subD"] * OUT["subZ"] * (C["T0"] - OUT["subT"]) / C["Lm"]
        c2 = OUT["subZ"] * (1 - OUT["subD"] / C["Dice"]) * C["Dice"]
        Wlim = np.minimum(c1, c2)

        # Refreezing of slush:
        # RS = min(subS, Wlim) where layer is cold and wet, else 0
        # No temporaries needed, min() handles both branches; mask zeroes inactive elements
        layer_cond = (OUT["subS"] > 0) & (OUT["subT"] < C["T0"])
        RS = np.minimum(OUT["subS"], Wlim)
        RS[~layer_cond] = 0.0

        # Update slush water content
        OUT["subS"] -= RS

        # Update temperature after refreezing
        OUT["subT"] += (C["Lm"] * RS) / (OUT["subD"] * OUT["cpi"] * OUT["subZ"])

        # Update density after refreezing
        OUT["subD"] += RS / OUT["subZ"]

        ###########################################################
        # REFREEZING OF IRREDUCIBLE WATER
        ###########################################################
        # Determine whether cold content or density limits the amount of refreezing
        OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]
        c1 = OUT["cpi"] * OUT["subD"] * OUT["subZ"] * (C["T0"] - OUT["subT"]) / C["Lm"]
        c2 = OUT["subZ"] * (1 - OUT["subD"] / C["Dice"]) * C["Dice"]
        Wlim = np.minimum(c1, c2)

        # Calculate refreezing amounts
        valid_mask = (OUT["subW"] > 0) & (OUT["subT"] < C["T0"])
        RI = np.minimum(OUT["subW"], Wlim)
        RI[~valid_mask] = 0.0

        # Update water content (subW), temperature (subT), and density (subD)
        OUT["subW"] -= RI
        OUT["subT"] += (C["Lm"] * RI) / (OUT["subD"] * OUT["cpi"] * OUT["subZ"])
        OUT["subD"] += RI / OUT["subZ"]

        # Determine total refreezing and individual contributions, and total slush water and irreducible water
        OUT["refr"] = 1e-3 * (np.sum(RP, axis=1) + np.sum(RS, axis=1) + np.sum(RI, axis=1))
        OUT["refr_P"] = 1e-3 * np.sum(RP, axis=1)  # Refreezing of percolating water
        OUT["refr_S"] = 1e-3 * np.sum(RS, axis=1)  # Refreezing of slush water
        OUT["refr_I"] = 1e-3 * np.sum(RI, axis=1)  # Refreezing of irreducible water
        OUT["slushw"] = np.sum(OUT["subS"], axis=1)  # Total stored slush water
        OUT["irrw"] = np.sum(OUT["subW"], axis=1)  # Total stored irreducible water

        return True

    def layer_merging_and_splitting():
        """
        Layer merging and splitting
        """
        if grid["doubledepth"]:
            subZ_old = OUT["subZ"].copy()
            subD_old = OUT["subD"].copy()
            subW_old = OUT["subW"].copy()
            subT_old = OUT["subT"].copy()
            subS_old = OUT["subS"].copy()
            for n in range(len(grid["split"])):  # Iterate through split points
                split = grid["split"][n] - 1

                # Merge Layers (Accumulation Case)
                cond_merge = (OUT["subZ"][:, split] <= (2.0**n) * grid["max_subZ"]) & (grid["mask"] == 1)

                # Update merged layers
                OUT["subZ"][cond_merge, split - 1] = subZ_old[cond_merge, split - 1] + subZ_old[cond_merge, split]
                OUT["subW"][cond_merge, split - 1] = subW_old[cond_merge, split - 1] + subW_old[cond_merge, split]
                OUT["subS"][cond_merge, split - 1] = subS_old[cond_merge, split - 1] + subS_old[cond_merge, split]
                OUT["subD"][cond_merge, split - 1] = (
                    subZ_old[cond_merge, split - 1] * subD_old[cond_merge, split - 1]
                    + subZ_old[cond_merge, split] * subD_old[cond_merge, split]
                ) / (subZ_old[cond_merge, split - 1] + subZ_old[cond_merge, split])
                OUT["subT"][cond_merge, split - 1] = (
                    subZ_old[cond_merge, split - 1] * subT_old[cond_merge, split - 1]
                    + subZ_old[cond_merge, split] * subT_old[cond_merge, split]
                ) / (subZ_old[cond_merge, split - 1] + subZ_old[cond_merge, split])

                # Shift properties up for merged layers
                OUT["subZ"][cond_merge, split:-1] = subZ_old[cond_merge, split + 1 :]
                OUT["subW"][cond_merge, split:-1] = subW_old[cond_merge, split + 1 :]
                OUT["subS"][cond_merge, split:-1] = subS_old[cond_merge, split + 1 :]
                OUT["subD"][cond_merge, split:-1] = subD_old[cond_merge, split + 1 :]
                OUT["subT"][cond_merge, split:-1] = subT_old[cond_merge, split + 1 :]

                # Adjust the newly added layer at the top
                OUT["subZ"][cond_merge, -1] = 2.0 ** len(grid["split"]) * grid["max_subZ"]
                OUT["subT"][cond_merge, -1] = 2.0 * subT_old[cond_merge, -1] - subT_old[cond_merge, -2]
                OUT["subD"][cond_merge, -1] = subD_old[cond_merge, -1]
                OUT["subW"][cond_merge, -1] = 0.0
                OUT["subS"][cond_merge, -1] = 0.0

                # Split Layers (Ablation Case)
                cond_split = (OUT["subZ"][:, split - 2] > (2.0**n) * grid["max_subZ"]) & (grid["mask"] == 1)

                # Update split layers
                OUT["subZ"][cond_split, split - 2] *= 0.5
                OUT["subW"][cond_split, split - 2] *= 0.5
                OUT["subS"][cond_split, split - 2] *= 0.5
                OUT["subT"][cond_split, split - 2] = subT_old[cond_split, split - 2]
                OUT["subD"][cond_split, split - 2] = subD_old[cond_split, split - 2]

                OUT["subZ"][cond_split, split - 1] = OUT["subZ"][cond_split, split - 2]
                OUT["subW"][cond_split, split - 1] = OUT["subW"][cond_split, split - 2]
                OUT["subS"][cond_split, split - 1] = OUT["subS"][cond_split, split - 2]
                OUT["subT"][cond_split, split - 1] = OUT["subT"][cond_split, split - 2]
                OUT["subD"][cond_split, split - 1] = OUT["subD"][cond_split, split - 2]

                # Shift properties down for split layers
                OUT["subZ"][cond_split, split:-1] = subZ_old[cond_split, split - 1 : -2]
                OUT["subW"][cond_split, split:-1] = subW_old[cond_split, split - 1 : -2]
                OUT["subS"][cond_split, split:-1] = subS_old[cond_split, split - 1 : -2]
                OUT["subT"][cond_split, split:-1] = subT_old[cond_split, split - 1 : -2]
                OUT["subD"][cond_split, split:-1] = subD_old[cond_split, split - 1 : -2]

                # Update runoff contributions
                OUT["runoff_irr_deep"][cond_split] += subW_old[cond_split, -1]
                OUT["runoff_slush"][cond_split] += subS_old[cond_split, -1]

        return True

    def runoff():
        ###########################################
        # RUNOFF
        ###########################################

        # Update runoff of irreducible water below the model bottom
        OUT["runoff_irr_deep_mean"] = OUT["runoff_irr_deep_mean"] * (1 - dt / C["yeardays"]) + OUT[
            "runoff_irr_deep"
        ] * (dt / C["yeardays"])

        # Calculate total runoff [in meters water equivalent per timestep]
        OUT["runoff"] = 1e-3 * (
            OUT["runoff_surface"] + OUT["runoff_slush"] + OUT["runoff_irr"] + OUT["runoff_irr_deep_mean"]
        )

        # Surface runoff [in meters water equivalent per timestep]
        OUT["runoff_surf"] = 1e-3 * OUT["runoff_surface"]

        # Slush runoff [in meters water equivalent per timestep]
        OUT["runoff_slush"] = 1e-3 * OUT["runoff_slush"]

        # Irreducible water runoff within the domain [in meters water equivalent per timestep]
        OUT["runoff_irr"] = 1e-3 * OUT["runoff_irr"]

        # Irreducible water runoff below the base of the domain [in meters water equivalent per timestep]
        OUT["runoff_irr_deep"] = 1e-3 * OUT["runoff_irr_deep_mean"]

        return True

    snowfall_and_deposition()
    melt_sublimation()
    compaction()
    heat_conduction()
    percolation_refreezing_and_storage()
    layer_merging_and_splitting()
    runoff()
    OUT["T_ice"] = OUT["subT"][:, -1]

    return OUT
