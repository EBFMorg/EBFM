import numpy as np

from coupling import Coupler


def main(C, OUT, IN, grid, cpl: Coupler) -> tuple[np.ndarray, dict]:
    """
    Computes the incoming shortwave radiation, including direct, diffuse,
    and total radiation after shading and atmospheric transmissivity.

    Parameters:
        C (dict): Constants used in the model, including atmospheric parameters.
        OUT (dict): Dictionary containing existing computed values, like `TOA`.
        IN (dict): Input data, including cloud cover (C), pressure (Pres), specific humidity (q), etc.
        grid (dict): Grid information, including latitude.
        cpl (Coupler): Coupling object for data exchange with external models.

    Returns:
        tuple: SWin (numpy array of incoming shortwave radiation) and updated OUT dictionary.
    """

    if cpl.couple_to_icon_atmo:
        SWin_diff = (0.8 - 0.65 * (1 - IN["C"])) * IN['SWin']
        SWin_dir = (0.2 + 0.65 * (1 - IN["C"])) * (1 - OUT["shade"]) * IN['SWin']
        SWin = SWin_dir + SWin_diff

    else:
        ###########################################################
        # Top of the atmosphere radiation
        ###########################################################
        lat_rad = OUT['lat_rad']
        d_rad = OUT['d_rad']
        h_rad = OUT['h_rad']

        # On a flat surface
        OUT["TOAflat"] = OUT["I0"] * (
                np.sin(lat_rad) * np.sin(d_rad) +
                np.cos(lat_rad) * np.cos(d_rad) * np.cos(h_rad)
        )  # SOURCE: Iqbal (1983)

        # On an inclined surface
        slopebeta = grid["slope_beta"]  # Slope angle
        slopegamma = grid["slope_gamma"]  # Slope azimuth angle
        OUT["TOA"] = OUT["I0"] * (
                (np.sin(lat_rad) * np.cos(slopebeta) - np.cos(lat_rad) * np.sin(slopebeta) * np.cos(
                    slopegamma)) * np.sin(
            d_rad) +
                (np.cos(lat_rad) * np.cos(slopebeta) + np.sin(lat_rad) * np.sin(slopebeta) * np.cos(
                    slopegamma)) * np.cos(
            d_rad) * np.cos(h_rad) +
                np.cos(d_rad) * np.sin(slopebeta) * np.sin(slopegamma) * np.sin(h_rad)
        )
        OUT["TOA"] = np.maximum(OUT["TOA"], 0.0)
        ###########################################################
        # Direct, diffuse and total radiation after shading
        ###########################################################

        # Calculation of direct and diffuse radiation after shading
        OUT["TOAdir"] = (0.2 + 0.65 * (1 - IN["C"])) * (1 - OUT["shade"]) * OUT["TOA"]  # SOURCE: Oerlemans (1992)
        OUT["TOAdiff"] = (0.8 - 0.65 * (1 - IN["C"])) * OUT["TOA"]
        OUT["TOAshade"] = OUT["TOAdir"] + OUT["TOAdiff"]


        ###########################################################
        # Atmospheric transmissivity
        ###########################################################

        # Transmissivity after gaseous absorption / scattering
        m = 35.0 * (IN["Pres"] / C["Pref"]) * (
                1224.0 * (OUT["TOAflat"] / OUT["I0"]) ** 2 + 1.0) ** -0.5  # SOURCE: Meyers and Dale (1983)
        t_rg = 1.021 - 0.084 * np.sqrt(
            m * (949.0 * (IN["Pres"] / 1e3) * 1e-5 + 0.051))  # SOURCE: Atwater and Brown Jr (1974)

        # Transmissivity after water vapor absorption
        temp_dew_kelvin = (1 / 273.15 - (C["Rv"] / C["Ls"]) * np.log(
            IN["q"] * IN["Pres"] / (C["eps"] * C["VP0"])
        )) ** -1  # SOURCE: McDonald (1960)
        temp_dew_fahr = 32.0 + 1.8 * (temp_dew_kelvin - 273.15)

        # Assign lambda based on latitude region
        lat_mean_abs = np.abs(np.mean(grid["lat"]))
        if lat_mean_abs < 20:
            lambda_value = 2.91
        elif lat_mean_abs < 30:
            lambda_value = 3.12
        elif lat_mean_abs < 40:
            lambda_value = 3.00
        elif lat_mean_abs < 50:
            lambda_value = 2.78
        elif lat_mean_abs < 60:
            lambda_value = 2.79
        elif lat_mean_abs < 70:
            lambda_value = 2.41
        elif lat_mean_abs < 80:
            lambda_value = 2.03
        else:
            lambda_value = 1.62

        # Water vapor path
        u = np.exp(0.1133 - np.log(lambda_value + 1.0) + 0.0393 * temp_dew_fahr)
        t_w = 1 - 0.077 * (u * m) ** 0.3  # Transmissivity after water vapor absorption

        # Transmissivity after aerosol absorption
        t_a = C["k_aer"] ** m  # SOURCE: Houghton (1954)

        # Transmissivity after cloud absorption / scattering
        t_cl = 1.0 - 0.128 * IN["C"] - 0.346 * IN["C"] ** 2  # SOURCE: Van Pelt et al. (2012)

        ###########################################################
        # Incoming solar radiation
        ###########################################################

        SWin = OUT["TOAshade"] * t_rg * t_w * t_a * t_cl

    return SWin, OUT
