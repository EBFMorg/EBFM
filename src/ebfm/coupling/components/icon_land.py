# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component, ExchangeKeySet

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, Timestep
from ebfm.core.config import ComponentId
from ebfm.core.constants import LATENT_HEAT_OF_FUSION, MELTING_POINT, WATER_DENSITY


def partition_evapotrans(evapotrans: np.ndarray, Tsurf: np.ndarray, melt: np.ndarray) -> dict[str, np.ndarray]:
    """
    Partition the evapotranspiration received from ICON-Land into EBFM's moisture terms.

    Follows the logic of LOOP_EBM: for a surface below the melting point the latent heat
    exchange is sublimation (loss) or deposition (gain), at the melting point evaporation or
    condensation. Evaporation is limited by the available melt water as in LOOP_EBM.

    @param[in] evapotrans evapotranspiration incl. sublimation in m w.e. per time step, negative upward (loss)
    @param[in] Tsurf surface temperature (K)
    @param[in] melt melt in m w.e. per time step

    @returns dictionary with moist_sublimation, moist_evaporation (losses, positive),
             moist_deposition, moist_condensation (gains, positive), all in m w.e. per time step
    """
    frozen = Tsurf < MELTING_POINT
    loss = np.maximum(-evapotrans, 0.0)
    gain = np.maximum(evapotrans, 0.0)

    moist_sublimation = np.where(frozen, loss, 0.0)
    moist_evaporation = np.where(~frozen, loss, 0.0)
    moist_evaporation = np.minimum(moist_evaporation, melt)
    moist_deposition = np.where(frozen, gain, 0.0)
    moist_condensation = np.where(~frozen, gain, 0.0)

    return {
        "moist_sublimation": moist_sublimation,
        "moist_evaporation": moist_evaporation,
        "moist_deposition": moist_deposition,
        "moist_condensation": moist_condensation,
    }


class IconLand(Component):
    """
    Component class for coupling to the ICON land model (ICON-Land / JSBACH).

    The coupling to ICON-Land is separate from the coupling to the ICON atmosphere: ICON-Land
    is its own YAC component (``icon-land``) living on the ICON atmosphere processes.

    EBFM sends its surface and firn state (ice fraction, surface albedo, first firn layer
    temperature, conductance and heat capacity, runoff, surface mass balance, snow mass) to ICON-Land
    and receives the results of the surface energy balance computed by ICON-Land (JSBACH) on its
    glacier tile, averaged over the EBFM time step: surface temperature, melt and evapotranspiration.
    map_energy_balance_to_ebfm turns those into EBFM's output variables, which replace the results of
    EBFM's own surface energy balance and drive its snow/firn model and mass balance.

    The results depend on the state, so the two are communicated in separate phases: surface_state,
    then (once ICON-Land has computed the surface energy balance from it) energy_balance. This lets
    the caller send the state, do other independent work (e.g. exchange with icon_atmo) while
    ICON-Land computes, and only then receive the results, instead of blocking on ICON-Land right
    after sending to it.
    """

    surface_state = ExchangeKeySet(
        name="surface state",
        source_keys={
            "icefract",
            "albedo",
            "t_sub",
            "ghf_cond",
            "hcap_sub",
            "runoff_to_icon_land",
            "smb_to_icon_land",
            "snowmass",
        },
    )
    energy_balance = ExchangeKeySet(
        name="surface energy balance",
        target_keys={"t_srf", "melt", "evapotrans"},
    )
    accepted_exchange_key_sets = (surface_state, energy_balance)

    def __init__(self, coupler: "Coupler", name: str = ComponentId.ICON_LAND.value):
        super().__init__(coupler, name)

    def get_field_definitions(self) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to IconLand.
        """
        timestep = Timestep(value=self.ebfm_time.time_step_iso8601())

        return FieldSet(
            {
                Field(
                    name="icefract",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Ice-covered fraction of the EBFM grid cell (1: glacier, 0: no glacier)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="albedo",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface albedo of the EBFM grid cell (fraction)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="t_sub",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Temperature of the first subsurface firn layer (K)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="ghf_cond",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Conductance between surface and first subsurface firn layer (W m-2 K-1)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="hcap_sub",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Heat capacity of the firn surface layer (J m-2 K-1)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="runoff_to_icon_land",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Runoff from the firn column (kg m-2 s-1)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="smb_to_icon_land",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Climatic surface mass balance (kg m-2 s-1)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="snowmass",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Snow mass on top of the ice (kg m-2)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                # Fields received from ICON-Land (glacier tile, averaged over the EBFM time step)
                Field(
                    name="t_srf",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface temperature of the glacier tile (K)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="melt",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Snow/ice melt on the glacier tile (kg m-2 s-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="evapotrans",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Evapotranspiration incl. sublimation, negative upward (kg m-2 s-1)",
                    exchange_type=ExchangeType.TARGET,
                ),
            }
        )

    def _exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray],
        requested_key_set: ExchangeKeySet,
    ) -> dict[str, np.ndarray]:
        """
        Send the surface/firn state to IconLand, or receive the surface energy balance results,
        depending on the requested key set.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values Mapping of field names to fallback values to use if get fails
        @param[in] requested_key_set key set to be communicated: surface_state or energy_balance

        @returns for energy_balance, a dictionary of received field data: "t_srf" (K), "melt" and
                 "evapotrans" (m w.e. per EBFM time step, evapotrans negative upward), only the fields
                 that are actually coupled; empty for surface_state
        """
        received_data: dict[str, np.ndarray] = {}

        if requested_key_set == self.surface_state:
            from ebfm.core.constants import WATER_DENSITY

            self._put_if_coupled("icefract", data_to_exchange)
            self._put_if_coupled("albedo", data_to_exchange)
            self._put_if_coupled("t_sub", data_to_exchange)
            self._put_if_coupled("ghf_cond", data_to_exchange)
            self._put_if_coupled("hcap_sub", data_to_exchange)
            self._put_if_coupled("runoff_to_icon_land", data_to_exchange, transform=self._map_mass_flux_from_ebfm)
            self._put_if_coupled("smb_to_icon_land", data_to_exchange, transform=self._map_mass_flux_from_ebfm)
            self._put_if_coupled("snowmass", data_to_exchange, transform=lambda x: x * WATER_DENSITY)
        elif requested_key_set == self.energy_balance:
            t_srf = self._get_if_coupled("t_srf", fallback_values=fallback_values)
            if t_srf is not None:
                received_data["t_srf"] = t_srf

            for name in ("melt", "evapotrans"):
                data = self._get_if_coupled(
                    name, transform=self._map_mass_flux_to_ebfm, fallback_values=fallback_values
                )
                if data is not None:
                    received_data[name] = data
        else:
            raise RuntimeError(f"Unexpected {requested_key_set=}")

        return received_data

    def map_energy_balance_to_ebfm(self, data_from_icon_land: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Map the surface energy balance received from ICON-Land to EBFM's output variables.

        They replace the results of EBFM's own surface energy balance, which is kept as a
        diagnostic. The radiative and turbulent fluxes are not part of this: EBFM keeps reporting
        its own; JSBACH's fluxes are available in the ICON-Land output.

        @param[in] data_from_icon_land fields received in an energy_balance exchange: t_srf (K),
                                       melt and evapotrans (m w.e. per time step, evapotrans
                                       negative upward)

        @returns Tsurf (K), melt and the moist_* terms (m w.e. per time step) and Emelt (W m-2)

        @raises RuntimeError if a field of the surface energy balance has not been received
        """
        missing = sorted(self.energy_balance.target_keys - set(data_from_icon_land))
        if missing:
            raise RuntimeError(
                f"Coupled to ICON-Land, but the fields {missing} of its surface energy balance have not been "
                "received. They are required to run the firn model, so check that they are coupled in the "
                "coupler configuration and that ICON-Land provides them in every EBFM time step."
            )

        # Surface temperature: JSBACH limits the glacier surface temperature to the melting point;
        # the time average may not exceed it either, but guard against round-off.
        Tsurf = np.minimum(np.asarray(data_from_icon_land["t_srf"], dtype=float), MELTING_POINT)

        # Melt (m w.e. per time step, converted from the received mass flux in _exchange)
        melt = np.maximum(np.asarray(data_from_icon_land["melt"], dtype=float), 0.0)

        # Moisture terms from the evapotranspiration (m w.e. per time step, negative upward)
        moist = partition_evapotrans(np.asarray(data_from_icon_land["evapotrans"], dtype=float), Tsurf, melt)

        # Energy equivalent of the melt (W m-2), for output only
        seconds_per_timestep = self.ebfm_time.time_step_in_seconds()
        Emelt = melt * WATER_DENSITY * LATENT_HEAT_OF_FUSION / seconds_per_timestep

        return {"Tsurf": Tsurf, "melt": melt, "Emelt": Emelt, **moist}
