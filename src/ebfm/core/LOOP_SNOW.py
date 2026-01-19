# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ebfm.constants import SECONDS_PER_HOUR


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

    def percolation_refreezing_and_storage():
        #########################################################
        # Percolation, refreezing and irreducible water storage
        #########################################################
        subW_old = OUT["subW"].copy()  # Store the old water content
        gpsum, nl = OUT["subT"].shape

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
        cond1 = c1 >= c2  # No need for separate cond2 as it's just the negation

        # Compute refreezing potential (`Wlim`) per layer
        Wlim = np.maximum(np.where(cond1, c2, c1), 0)

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
        RP = np.zeros_like(OUT["subZ"])  # Refreezing potential
        leftW = np.zeros(grid["gpsum"])  # Remaining water
        avail_W_loc = np.zeros(grid["gpsum"])  # Available water per layer

        for n in range(nl):
            # Compute available water per layer
            avail_W_loc += carrot[:, n]

            # Condition: water > refreezing limit
            cond1 = avail_W_loc > Wlim[:, n]

            RP[cond1, n] = Wlim[cond1, n]  # Refreezing limited by Wlim
            leftW[cond1] = avail_W_loc[cond1] - Wlim[cond1, n]
            OUT["subW"][cond1, n] = subW_old[cond1, n] + np.minimum(leftW[cond1], Wirr[cond1, n])

            # Condition: water <= refreezing limit
            RP[~cond1, n] = avail_W_loc[~cond1]
            OUT["subW"][~cond1, n] = subW_old[~cond1, n]

            # Update available water after refreezing
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

        # Initialize slush water in all layers
        OUT["subS"] = np.zeros((grid["gpsum"], nl))

        # Bottom-up filling of pore space with slush water
        for n in range(nl - 1, -1, -1):  # Loop from bottom (nl) to top (1)
            cond1 = avail_S > slushspace[:, n]  # Mask for layers where more water is available than the slush space
            OUT["subS"][cond1, n] = slushspace[cond1, n]  # Fill slush space in those layers
            OUT["subS"][~cond1, n] = avail_S[~cond1]  # Fill remaining water in other layers
            avail_S -= OUT["subS"][:, n]  # Reduce available slush water by the amount stored

        #####################################
        # Refreezing of slush water
        #####################################
        # Determine whether cold content or density limits the amount of refreezing
        OUT["cpi"] = 152.2 + 7.122 * OUT["subT"]
        c1 = OUT["cpi"] * OUT["subD"] * OUT["subZ"] * (C["T0"] - OUT["subT"]) / C["Lm"]
        c2 = OUT["subZ"] * (1 - OUT["subD"] / C["Dice"]) * C["Dice"]
        Wlim = np.minimum(c1, c2)

        # Available slush water
        slush_W = OUT["subS"].copy()  # Make a copy to avoid overwriting inputs (vectorized across all layers)

        # Determine refreezing amounts
        layer_cond = (OUT["subS"] > 0) & (OUT["subT"] < C["T0"])
        Wlim_effective = Wlim * layer_cond
        slush_W_effective = slush_W * layer_cond
        RS = np.where(slush_W_effective > Wlim_effective, Wlim_effective, slush_W_effective)

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
        Wlim = np.where(c1 >= c2, c2, c1)

        # Calculate refreezing amounts
        valid_mask = (OUT["subW"] > 0) & (OUT["subT"] < C["T0"])
        RI = np.minimum(OUT["subW"], Wlim) * valid_mask

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
