# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray

from .compute_backend import get_backend, ComputeBackend
from .constants import SECONDS_PER_HOUR
from .LOOP_SNOW_kernels import (
    _compaction_kernel,
    _heat_conduction_kernel,
    _heat_conduction_prep_kernel,
    _percolation_kernel,
)

# SnowDeviceState is imported lazily inside the GPU branch of main() so that
# NumPy/Numba-only users never trigger the numba.cuda / numba.hip import (and
# its import-time vendor detection) in LOOP_SNOW_gpu_kernels.

# line_profiler support: `profile` is injected as a builtin by kernprof.
# When running normally, fall back to a no-op so the decorator stays in place.
try:
    profile  # noqa: F821
except NameError:
    profile = lambda f: f  # noqa: E731

from ebfm.core import logging

_SUCCESS = True

# Layer thicknesses (m) below this count as zero, i.e. the layer is melted away.
# Adapted from initial commit 42c0113
_MIN_LAYER_THICKNESS = 1e-17

logger = logging.getLogger(__name__)


def main(C, OUT, IN, dt: float, grid, phys):
    """
    Implementation of the multi-layer snow and firn model.

    Each glacier grid point (total number of grid points `grid["gpsum"]`) carries one vertical snow/firn "column" of
    `grid["nl"]` layers. Per-column state (subT, subD, subZ, subW, subS, ...) is stored in OUT as 2D arrays of shape
    (grid["gpsum"], grid["nl"]): axis 0 indexes the column (grid point), axis 1 indexes the layer within that column,
    from layer 0 (a thin surface ghost layer, not physical) down to the deepest layer. Columns are independent and
    updated in parallel, either vectorized over axis 0 (NumPy path) or via Numba `prange` over columns (Numba path).
    `dt` is the global time step shared by all columns; some steps (see heat_conduction) locally sub-step individual
    columns with their own CFL-limited dt_local until every column has caught up to `dt`.

    Parameters:
        C (dict): Model constants and parameters.
        OUT (dict): Output variables to store results.
        IN (dict): Input data for the model.
        dt (float): Global model time-step, shared by all columns.
        grid (dict): Grid geometry, incl. gpsum (number of grid points/columns), nl (layers per column), max_subZ (max.
            top-layer thickness), and doubledepth/split (layer-merging/splitting thresholds,see
            layer_merging_and_splitting).
        phys (dict): Model physics settings.

    Returns:
        dict: Updated OUT dictionary.
    """

    logger.debug("Starting LOOP_SNOW...")

    @profile
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
            # Fresh snow thickness (m) deposited in this pass, at most one full layer
            shift_amount = np.minimum(shift_tot, max_subZ)
            shift_tot -= shift_amount

            # No-shift branch only touches top layer, only column 0 copied (instead of whole grid).
            subZ_top_old = OUT["subZ"][:, 0].copy()
            subT_top_old = OUT["subT"][:, 0].copy()
            subD_top_old = OUT["subD"][:, 0].copy()

            # A "grid shift" moves the layer indices: an overfull top layer becomes
            # layer 1, deeper layers move down, the overflow starts a new top layer.
            no_grid_shift: NDArray[np.bool_] = subZ_top_old + shift_amount <= max_subZ
            needs_grid_shift: NDArray[np.bool_] = ~no_grid_shift
            # Index into the grid with indices, not with the masks, reason: NumPy
            # re-scans the full mask on every indexing operation, flatnonzero pays that
            # scan once and each later access then costs only the affected rows.
            idx_noshift: NDArray[np.intp] = np.flatnonzero(no_grid_shift)
            idx_shift: NDArray[np.intp] = np.flatnonzero(needs_grid_shift)

            # Handle no-shift updates (vectorized)
            OUT["subZ"][idx_noshift, 0] += shift_amount[idx_noshift]
            subZ_top_new = OUT["subZ"][idx_noshift, 0]
            OUT["subT"][idx_noshift, 0] = (
                subT_top_old[idx_noshift] * subZ_top_old[idx_noshift] / subZ_top_new
                + OUT["Tsurf"][idx_noshift] * shift_amount[idx_noshift] / subZ_top_new
            )
            OUT["subD"][idx_noshift, 0] = (
                subD_top_old[idx_noshift] * subZ_top_old[idx_noshift] / subZ_top_new
                + OUT["Dfreshsnow"][idx_noshift] * shift_amount[idx_noshift] / subZ_top_new
            )

            # Handle shifting updates (vectorized). Shift branch needs full
            # rows, so copy only the (usually few) overflowing columns.
            if idx_shift.size:
                subZ_old = OUT["subZ"][idx_shift]
                subT_old = OUT["subT"][idx_shift]
                subD_old = OUT["subD"][idx_shift]
                subW_old = OUT["subW"][idx_shift]

                OUT["subZ"][idx_shift, 2 : nl - 1] = subZ_old[:, 1 : nl - 2]
                OUT["subT"][idx_shift, 2 : nl - 1] = subT_old[:, 1 : nl - 2]
                OUT["subD"][idx_shift, 2 : nl - 1] = subD_old[:, 1 : nl - 2]
                OUT["subW"][idx_shift, 2 : nl - 1] = subW_old[:, 1 : nl - 2]

                OUT["subZ"][idx_shift, 1] = max_subZ
                OUT["subZ"][idx_shift, 0] = (subZ_old[:, 0] + shift_amount[idx_shift]) - max_subZ
                subZ_layer1_new = OUT["subZ"][idx_shift, 1]
                OUT["subT"][idx_shift, 1] = (
                    subT_old[:, 0] * subZ_old[:, 0] / subZ_layer1_new
                    + OUT["Tsurf"][idx_shift] * (subZ_layer1_new - subZ_old[:, 0]) / subZ_layer1_new
                )
                OUT["subT"][idx_shift, 0] = OUT["Tsurf"][idx_shift]
                OUT["subD"][idx_shift, 1] = (
                    subD_old[:, 0] * subZ_old[:, 0] / subZ_layer1_new
                    + OUT["Dfreshsnow"][idx_shift] * (subZ_layer1_new - subZ_old[:, 0]) / subZ_layer1_new
                )
                OUT["subD"][idx_shift, 0] = OUT["Dfreshsnow"][idx_shift]
                OUT["subW"][idx_shift, 1] = subW_old[:, 0]
                OUT["subW"][idx_shift, 0] = 0.0

                # Update runoff for shifted layers
                OUT["runoff_irr_deep"][idx_shift] += subW_old[:, nl - 1]

        return _SUCCESS

    @profile
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
            # Thickness (m, negative) removed in this pass, at most the top two layers
            shift_amount = np.maximum(shift_tot, -OUT["subZ"][:, 1])
            shift_tot -= shift_amount

            OUT["surfH"] += shift_amount

            nl = grid["nl"]

            # No-shift branch only touches top layer, only column 0 copied (instead of whole grid).
            subZ_top_old = OUT["subZ"][:, 0].copy()
            subT_top_old = OUT["subT"][:, 0].copy()
            subD_top_old = OUT["subD"][:, 0].copy()
            subW_top_old = OUT["subW"][:, 0].copy()

            # A "grid shift" moves the layer indices: a fully melted top layer is
            # dropped, layer 1 becomes the new top layer, deeper layers move up.
            # See snowfall_and_deposition() for explanation of the indexing strategy.
            no_grid_shift: NDArray[np.bool_] = subZ_top_old + shift_amount > _MIN_LAYER_THICKNESS
            needs_grid_shift: NDArray[np.bool_] = ~no_grid_shift
            idx_noshift: NDArray[np.intp] = np.flatnonzero(no_grid_shift)
            idx_shift: NDArray[np.intp] = np.flatnonzero(needs_grid_shift)

            # Handle the no-shift case
            subZ_top_new = subZ_top_old[idx_noshift] + shift_amount[idx_noshift]
            OUT["subZ"][idx_noshift, 0] = subZ_top_new
            OUT["subT"][idx_noshift, 0] = subT_top_old[idx_noshift]
            OUT["subD"][idx_noshift, 0] = subD_top_old[idx_noshift]
            # Liquid water scales with the remaining fraction of the top layer
            OUT["subW"][idx_noshift, 0] = subW_top_old[idx_noshift] * (subZ_top_new / subZ_top_old[idx_noshift])

            # Handle the shift case: copy only the shift rows.
            if idx_shift.size:
                subZ_old = OUT["subZ"][idx_shift]
                subT_old = OUT["subT"][idx_shift]
                subD_old = OUT["subD"][idx_shift]
                subW_old = OUT["subW"][idx_shift]

                OUT["subZ"][idx_shift, 1 : nl - 2] = subZ_old[:, 2 : nl - 1]
                OUT["subT"][idx_shift, 1 : nl - 2] = subT_old[:, 2 : nl - 1]
                OUT["subD"][idx_shift, 1 : nl - 2] = subD_old[:, 2 : nl - 1]
                OUT["subW"][idx_shift, 1 : nl - 2] = subW_old[:, 2 : nl - 1]

                subZ_top_new = subZ_old[:, 0] + subZ_old[:, 1] + shift_amount[idx_shift]
                OUT["subZ"][idx_shift, 0] = subZ_top_new
                OUT["subT"][idx_shift, 0] = subT_old[:, 1]
                OUT["subD"][idx_shift, 0] = subD_old[:, 1]
                # Liquid water scales with the remaining fraction of the former layer 1
                OUT["subW"][idx_shift, 0] = subW_old[:, 1] * (subZ_top_new / subZ_old[:, 1])
                OUT["subT"][idx_shift, nl - 1] = subT_old[:, nl - 1]
                OUT["subW"][idx_shift, nl - 1] = 0.0

                # Update the deepest layer properties (vectorized over shift rows)
                if grid["doubledepth"] == 1:
                    OUT["subZ"][idx_shift, nl - 1] = 2.0 ** len(grid["split"]) * grid["max_subZ"]
                else:
                    OUT["subZ"][idx_shift, nl - 1] = grid["max_subZ"]
                OUT["subD"][idx_shift, nl - 1] = subD_old[:, nl - 1]

        return _SUCCESS

    @profile
    def compaction(gpu=None):
        """
        Calculate snow and firn compaction and update density and layer thickness
        """

        gpsum, nl = grid["gpsum"], grid["nl"]
        Dice, Dfirn = C["Dice"], C["Dfirn"]

        dt_yearfrac = dt / C["yeardays"]
        dt_seconds = dt * C["dayseconds"]

        # Pre-zero diagnostic arrays once
        # Avoids np.zeros_like allocation for each array on every timestep.
        _dshape = OUT["subD"].shape
        for _key in ("Dens_destr_metam", "Dens_overb_pres", "Dens_drift"):
            if _key not in OUT or OUT[_key].shape != _dshape:
                OUT[_key] = np.zeros(_dshape)
            else:
                OUT[_key].fill(0.0)

        # runoff_irr is written by the kernel; ensure it exists before it reads.
        _gshape = (gpsum,)
        if "runoff_irr" not in OUT or OUT["runoff_irr"].shape != _gshape:
            OUT["runoff_irr"] = np.zeros(_gshape)

        # Numba / GPU kernel path:
        backend = get_backend()
        if backend in (ComputeBackend.NUMBA, ComputeBackend.GPU):
            _mode = {"firn_only": 0, "firn+snow": 1}.get(phys["snow_compaction"], -1)
            if _mode < 0:
                raise ValueError(f"_compaction_kernel: unknown snow_compaction={phys['snow_compaction']!r}")
            _tau_drift = 48 * 2 * SECONDS_PER_HOUR
            if gpu is not None:
                # GPU: kernel launched on the resident device arrays; the
                # pre-compaction subD/subZ snapshot is taken on-device.
                gpu.compaction(
                    OUT["Dens_destr_metam"],
                    OUT["Dens_overb_pres"],
                    OUT["Dens_drift"],
                    dt_yearfrac,
                    dt_seconds,
                    dt,
                    C,
                    _tau_drift,
                    _mode,
                )
                return _SUCCESS

            _compaction_kernel(
                OUT["subD"],
                OUT["subZ"],
                OUT["subT"],
                OUT["subW"],
                OUT["subTmean"],
                IN["logyearsnow"],
                IN["yearsnow"],
                IN["WS"],
                OUT["Dens_destr_metam"],
                OUT["Dens_overb_pres"],
                OUT["Dens_drift"],
                OUT["surfH"],
                OUT["sumWinit"],
                OUT["runoff_irr"],
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
                _tau_drift,
                _mode,
            )
        else:
            # NumPy path
            subD_old = OUT["subD"].copy()
            subZ_old = OUT["subZ"].copy()
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

                # Set gravitational constants and masks
                # Use OUT["subD"] and IN["logyearsnow"] directly
                # cond_firn is already incorporated into both masks
                grav_const = np.zeros_like(OUT["subD"])
                low_density_mask = cond_firn & (OUT["subD"] < 550)
                high_density_mask = cond_firn & (OUT["subD"] >= 550)
                grav_const[low_density_mask] = 0.07 * np.maximum(
                    1.435 - 0.151 * IN["logyearsnow"][low_density_mask], 0.25
                )
                grav_const[high_density_mask] = 0.03 * np.maximum(
                    2.366 - 0.293 * IN["logyearsnow"][high_density_mask], 0.25
                )

                # Update firn densities
                temp_factor = np.exp(-C["Ec"] / (C["rd"] * OUT["subT"]) + C["Eg"] / (C["rd"] * OUT["subTmean"]))
                firn_increment = dt_yearfrac * grav_const * IN["yearsnow"] * C["g"] * (Dice - OUT["subD"]) * temp_factor
                np.add(OUT["subD"], firn_increment, where=cond_firn, out=OUT["subD"])
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
                np.add(OUT["subD"], snow_increment, where=cond_snow, out=OUT["subD"])
                np.minimum(OUT["subD"], Dice, where=cond_snow, out=OUT["subD"])

                # Store densification by destructive metamorphism
                np.copyto(OUT["Dens_destr_metam"], snow_increment, where=cond_snow)

                # ------ DENSIFICATION BY OVERBURDEN PRESSURE ------ #
                CC5, CC6 = 0.1, 0.023
                CC7 = (
                    4.0 * 7.62237e6 / 250.0 * OUT["subD"] * 1 / (1 + 60 * OUT["subW"] * 1 / (C["Dwater"] * OUT["subZ"]))
                )

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
                np.minimum(OUT["subD"], C["Dice"], where=cond_snow, out=OUT["subD"])

                # Store densification by overburden pressure
                OUT["Dens_overb_pres"][cond_snow] = (
                    dt * C["dayseconds"] * OUT["subD"][cond_snow] * Psload[cond_snow] / Visc[cond_snow]
                )

                # ------ DRIFTING SNOW DENSIFICATION ------ #
                MO = -0.069 + 0.66 * (1.25 - 0.0042 * (np.maximum(OUT["subD"], 50) - 50))
                SI = -2.868 * np.exp(-0.085 * IN["WS"][:, np.newaxis]) + 1 + MO
                cond_drift = SI > 0

                z_i = np.zeros_like(OUT["subZ"])
                if nl > 1:
                    z_i[:, 1:] = np.cumsum(OUT["subZ"][:, :-1] * (3.25 - SI[:, :-1]), axis=1)
                gamma_drift = np.maximum(0, SI * np.exp(-z_i / 0.1))
                tau = 48 * 2 * SECONDS_PER_HOUR
                with np.errstate(divide="ignore", invalid="ignore"):
                    tau_i = tau / gamma_drift

                # Update densities
                drift_increment = dt_seconds * np.maximum(350 - OUT["subD"], 0) / tau_i
                cond_drift_total = cond_drift & (OUT["subD"] < Dfirn)
                np.add(OUT["subD"], drift_increment, where=cond_drift_total, out=OUT["subD"])
                np.minimum(OUT["subD"], Dice, where=cond_drift_total, out=OUT["subD"])

                # Store densification by wind shearing
                np.copyto(OUT["Dens_drift"], drift_increment, where=cond_drift_total)

            # ------ UPDATE LAYER THICKNESS & SURFACE HEIGHT AFTER COMPACTION ------ #
            cond_layers = OUT["subD"] < Dice

            # Update layer thickness
            OUT["subZ"][cond_layers] = subZ_old[cond_layers] * subD_old[cond_layers] / OUT["subD"][cond_layers]

            # Update irreducible water storage: `mliqmax` is the maximum liquid water
            # (kg m-2) a layer holds against gravity; ice layers keep 0.0 and drain fully
            mliqmax = np.zeros((gpsum, nl))
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

        return _SUCCESS

    @profile
    def heat_conduction(gpu=None):
        """
        Calculate heat diffusion and update temperatures
        """
        if gpu is not None:
            # GPU: the prep / solve / boundary-clip kernels run on the resident
            # device arrays, so the host precompute and post-clip below are done
            # on-device (subK / subCeff are returned in download()).
            gpu.heat_conduction(dt, C)
            return _SUCCESS

        # ------ Heat Conduction Loop ------
        if get_backend() == ComputeBackend.NUMBA:
            # Numba parallel path
            # per-column precompute + CFL solve is done in _heat_conduction_prep_kernel
            gpsum, nl = OUT["subT"].shape
            kk = np.empty((gpsum, nl))
            c_eff = np.empty((gpsum, nl))
            kk_sz_top = np.empty(gpsum)
            kk_sz_interior = np.empty((gpsum, nl - 2))
            dz1 = np.empty(gpsum)
            dz2 = np.empty((gpsum, nl - 2))
            denom_layer1 = np.empty(gpsum)
            denom_interior = np.empty((gpsum, nl - 3))
            denom_bottom = np.empty(gpsum)
            dt_stab = np.empty(gpsum)
            _heat_conduction_prep_kernel(
                OUT["subD"],
                OUT["subZ"],
                OUT["subT"],
                kk,
                c_eff,
                kk_sz_top,
                kk_sz_interior,
                dz1,
                dz2,
                denom_layer1,
                denom_interior,
                denom_bottom,
                dt_stab,
                C["dayseconds"],
            )
            assert (dt_stab > 0).all(), "cells with dt_stab <= 0 are forbidden!"
            _heat_conduction_kernel(
                OUT["subT"],
                OUT["Tsurf"],
                kk_sz_top,
                kk_sz_interior,
                dz1,
                dz2,
                denom_layer1,
                denom_interior,
                denom_bottom,
                dt_stab,
                dt,
                C["dayseconds"],
                C["geothermal_flux"],
            )

        else:
            # NumPy path: host precompute + explicit while-loop with vectorized column updates. dt is the
            # global time step; each column uses local time stepping to advance from the current time until the
            # global dt is reached. The maximum local time step size of each sub-step is determined by the CFL
            # condition and the remainder until the end of the global time step.
            dz1 = (OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1]) ** 2
            dz2 = 0.5 * (OUT["subZ"][:, 2:] + OUT["subZ"][:, 1:-1]) ** 2
            kk = 0.138 - 1.01e-3 * OUT["subD"] + 3.233e-6 * OUT["subD"] ** 2  # Effective conductivity
            c_eff = OUT["subD"] * (152.2 + 7.122 * OUT["subT"])  # Effective heat capacity

            # Stability time step (CFL condition)
            # Layer 0: surface ghost layer, excluded from CFL condition
            dt_stab: NDArray[np.float64] = (
                0.5
                * np.min(c_eff[:, 1:], axis=1)
                * np.min(OUT["subZ"][:, 1:], axis=1) ** 2
                / np.max(kk[:, 1:], axis=1)
                / C["dayseconds"]
            )
            assert (dt_stab > 0).all(), "cells with dt_stab <= 0 are forbidden!"

            # Precompute kk*subZ products and the temperature-update denominators once
            # kk_sz_top: conductivity-thickness product for the top interface
            # kk_sz_interior: same for all interior interfaces
            kk_sz_top = kk[:, 0] * OUT["subZ"][:, 0] + 0.5 * kk[:, 1] * OUT["subZ"][:, 1]
            kk_sz_interior = kk[:, 1:-1] * OUT["subZ"][:, 1:-1] + kk[:, 2:] * OUT["subZ"][:, 2:]
            # denom_layer1: first active layer (layer 0: surface ghost layer overwritten from Tsurf)
            denom_layer1 = c_eff[:, 1] * (0.5 * OUT["subZ"][:, 0] + 0.5 * OUT["subZ"][:, 1] + 0.25 * OUT["subZ"][:, 2])
            denom_interior = c_eff[:, 2:-1] * (
                0.25 * OUT["subZ"][:, 1:-2] + 0.5 * OUT["subZ"][:, 2:-1] + 0.25 * OUT["subZ"][:, 3:]
            )
            denom_bottom = c_eff[:, -1] * (0.25 * OUT["subZ"][:, -2] + 0.75 * OUT["subZ"][:, -1])

            # Time elapsed for each column
            t_elapsed: NDArray[np.float64] = np.zeros(grid["gpsum"])
            # Vertical heat fluxes between layers
            kdTdz: NDArray[np.float64] = np.zeros_like(OUT["subT"])
            # Ping-pong buffers:
            # Pre-allocate two arrays once and swap references each iteration
            T_old = OUT["subT"].copy()
            T_new = np.empty_like(OUT["subT"])

            # Local time-stepping loop: advances each column by its own dt_local until all columns reach the end of the
            # global time step
            while np.any(t_elapsed < dt):
                # Copy T_old to T_new so inactive rows carry forward correctly across swaps
                np.copyto(T_new, T_old)

                dt_remainder = dt - t_elapsed  # determine remainder until end of global time step
                # use minimum of 1) maximum allowed dt due to stability and 2) remainder until end of global time step
                dt_local = np.minimum(dt_stab, dt_remainder)

                # Integer indices of still-active columns
                idx = np.flatnonzero(dt_local > 0)
                assert idx.size > 0, "no cells left where local time stepping has to be performed. Unexpected!"
                # Calculate vertical heat fluxes
                kdTdz[idx, 1] = kk_sz_top[idx] * (T_old[idx, 1] - OUT["Tsurf"][idx]) / dz1[idx]
                kdTdz[idx, 2:] = kk_sz_interior[idx] * (T_old[idx, 2:] - T_old[idx, 1:-1]) / dz2[idx]

                # Update layer-wise temperatures
                C_day_dt = C["dayseconds"] * dt_local[idx]

                T_new[idx, 1] = T_old[idx, 1] + C_day_dt * (kdTdz[idx, 2] - kdTdz[idx, 1]) / denom_layer1[idx]

                T_new[idx, 2:-1] = (
                    T_old[idx, 2:-1]
                    + C_day_dt[:, np.newaxis] * (kdTdz[idx, 3:] - kdTdz[idx, 2:-1]) / denom_interior[idx]
                )

                T_new[idx, -1] = T_old[idx, -1] + C_day_dt * (C["geothermal_flux"] - kdTdz[idx, -1]) / denom_bottom[idx]

                # Write final result back into the original OUT["subT"] array object in-place.
                np.copyto(OUT["subT"], T_new)

                # Swap buffer roles
                T_old, T_new = T_new, T_old

                # increase t_elapsed by local time step size
                t_elapsed += dt_local

            assert np.allclose(t_elapsed, dt), "some cells did not reach end of global time step!"

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

        return _SUCCESS

    @profile
    def percolation_refreezing_and_storage(gpu=None):
        #########################################################
        # Percolation, refreezing and irreducible water storage
        #########################################################
        gpsum, nl = OUT["subT"].shape

        backend = get_backend()
        if backend in (ComputeBackend.NUMBA, ComputeBackend.GPU):
            # Numba / GPU kernel path:
            _p_mode = {"bucket": 0, "normal": 1, "linear": 2, "uniform": 3}.get(phys["percolation"], -1)
            if _p_mode < 0:
                raise ValueError(f"_percolation_kernel: unknown percolation={phys['percolation']!r}")
            _avail_W = np.maximum(
                OUT["melt"] * 1e3 + IN["rain"] * 1e3 + (OUT["moist_condensation"] - OUT["moist_evaporation"]) * 1e3,
                0.0,
            )
            if gpu is not None:
                # GPU: kernel launched on the resident device arrays; the
                # post-heat subW snapshot is taken on-device.
                gpu.percolation(_avail_W, C, _p_mode, dt)
                return _SUCCESS

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
        else:
            # NumPy path
            subW_old = OUT["subW"].copy()  # Store the old water content
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
                OUT["subD"][noice]
                * OUT["subZ"][noice]
                * irr_factor
                * 0.05
                * np.minimum(C["Dice"] - OUT["subD"][noice], 20)
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
                # Layers down to the characteristic percolation depth (per column).
                # `ind` holds one index per column, so the range cannot be taken
                # with a slice, layers are selected with a mask instead.
                ind = np.argmin(np.abs(zz - z0), axis=1)
                carrot[np.arange(carrot.shape[1])[None, :] <= ind[:, None]] = 1 / z0
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

        return _SUCCESS

    @profile
    def layer_merging_and_splitting():
        """
        Layer merging and splitting
        """
        if not grid["doubledepth"]:
            return _SUCCESS

        # Precompute constants / reuse lookups
        max_subZ = grid["max_subZ"]
        mask1 = grid["mask"] == 1
        nsplit = len(grid["split"])
        top_thickness = (2.0**nsplit) * max_subZ

        for n in range(nsplit):  # Iterate through split points
            split = grid["split"][n]
            threshold = (2.0**n) * max_subZ

            # Merge Layers (Accumulation Case)
            idx_merge = np.flatnonzero((OUT["subZ"][:, split] <= threshold) & mask1)
            if idx_merge.size:
                # Snapshot only affected rows. Advanced indexing returns fresh
                # copy (holds pre-merge values). OUT written in place, without
                # necessity to copy whole (gpsum, nl) grid.
                subZ_old = OUT["subZ"][idx_merge]
                subD_old = OUT["subD"][idx_merge]
                subW_old = OUT["subW"][idx_merge]
                subT_old = OUT["subT"][idx_merge]
                subS_old = OUT["subS"][idx_merge]

                # Update merged layers
                OUT["subZ"][idx_merge, split - 1] = subZ_old[:, split - 1] + subZ_old[:, split]
                OUT["subW"][idx_merge, split - 1] = subW_old[:, split - 1] + subW_old[:, split]
                OUT["subS"][idx_merge, split - 1] = subS_old[:, split - 1] + subS_old[:, split]

                # Compute denominator once and reuse
                den = subZ_old[:, split - 1] + subZ_old[:, split]
                OUT["subD"][idx_merge, split - 1] = (
                    subZ_old[:, split - 1] * subD_old[:, split - 1] + subZ_old[:, split] * subD_old[:, split]
                ) / den
                OUT["subT"][idx_merge, split - 1] = (
                    subZ_old[:, split - 1] * subT_old[:, split - 1] + subZ_old[:, split] * subT_old[:, split]
                ) / den

                # Shift properties up for merged layers
                OUT["subZ"][idx_merge, split:-1] = subZ_old[:, split + 1 :]
                OUT["subW"][idx_merge, split:-1] = subW_old[:, split + 1 :]
                OUT["subS"][idx_merge, split:-1] = subS_old[:, split + 1 :]
                OUT["subD"][idx_merge, split:-1] = subD_old[:, split + 1 :]
                OUT["subT"][idx_merge, split:-1] = subT_old[:, split + 1 :]

                # Adjust the newly added layer at the base
                OUT["subZ"][idx_merge, -1] = top_thickness
                OUT["subT"][idx_merge, -1] = subT_old[:, -1]
                OUT["subD"][idx_merge, -1] = subD_old[:, -1]
                OUT["subW"][idx_merge, -1] = 0.0
                OUT["subS"][idx_merge, -1] = 0.0

            # Split Layers (Ablation Case). idx_split reflects the post-merge state.
            idx_split = np.flatnonzero((OUT["subZ"][:, split - 2] > threshold) & mask1)
            if idx_split.size:
                # Snapshot only the affected rows (post-merge values).
                subZ_old = OUT["subZ"][idx_split]
                subD_old = OUT["subD"][idx_split]
                subW_old = OUT["subW"][idx_split]
                subT_old = OUT["subT"][idx_split]
                subS_old = OUT["subS"][idx_split]

                # Update split layers
                OUT["subZ"][idx_split, split - 2] *= 0.5
                OUT["subW"][idx_split, split - 2] *= 0.5
                OUT["subS"][idx_split, split - 2] *= 0.5
                OUT["subT"][idx_split, split - 2] = subT_old[:, split - 2]
                OUT["subD"][idx_split, split - 2] = subD_old[:, split - 2]

                OUT["subZ"][idx_split, split - 1] = OUT["subZ"][idx_split, split - 2]
                OUT["subW"][idx_split, split - 1] = OUT["subW"][idx_split, split - 2]
                OUT["subS"][idx_split, split - 1] = OUT["subS"][idx_split, split - 2]
                OUT["subT"][idx_split, split - 1] = OUT["subT"][idx_split, split - 2]
                OUT["subD"][idx_split, split - 1] = OUT["subD"][idx_split, split - 2]

                # Shift properties down for split layers
                OUT["subZ"][idx_split, split:] = subZ_old[:, split - 1 : -1]
                OUT["subW"][idx_split, split:] = subW_old[:, split - 1 : -1]
                OUT["subS"][idx_split, split:] = subS_old[:, split - 1 : -1]
                OUT["subT"][idx_split, split:] = subT_old[:, split - 1 : -1]
                OUT["subD"][idx_split, split:] = subD_old[:, split - 1 : -1]

                # Update runoff contributions
                OUT["runoff_irr_deep"][idx_split] += subW_old[:, -1]
                OUT["runoff_slush"][idx_split] += subS_old[:, -1]

        return _SUCCESS

    @profile
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

        return _SUCCESS

    snowfall_and_deposition()
    melt_sublimation()
    if get_backend() == ComputeBackend.GPU:
        # Resident-state core: subT/subD/subZ/subW/subS stay on the device
        # across compaction -> heat conduction -> percolation. The subsurface
        # state is uploaded once here and downloaded once after percolation,
        # instead of round-tripping in each sub-step. The three functions run
        # their usual host-side glue and dispatch their kernel launches to the
        # device state; results match the NumPy/Numba paths.
        # Imported here (not at module top) so only GPU runs pull in numba.cuda.
        from .LOOP_SNOW_gpu_kernels import SnowDeviceState

        gpu_state = SnowDeviceState(
            OUT["subT"],
            OUT["subD"],
            OUT["subZ"],
            OUT["subW"],
            OUT["subS"],
            OUT["subTmean"],
            OUT["surfH"],
            IN["logyearsnow"],
            IN["yearsnow"],
            IN["WS"],
            OUT["Tsurf"],
            OUT["sumWinit"],
        )
        compaction(gpu_state)
        heat_conduction(gpu_state)
        percolation_refreezing_and_storage(gpu_state)
        gpu_state.download(OUT)
    else:
        compaction()
        heat_conduction()
        percolation_refreezing_and_storage()
    layer_merging_and_splitting()
    runoff()
    OUT["T_ice"] = OUT["subT"][:, -1]

    return OUT
