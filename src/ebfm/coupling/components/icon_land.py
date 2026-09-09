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
from ebfm.core.config import ComponentId, TimeConfig


class IconLand(Component):
    """
    Component class for coupling to the ICON land model (ICON-Land / JSBACH).

    The coupling to ICON-Land is separate from the coupling to the ICON atmosphere: ICON-Land
    is its own YAC component (``icon-land``) living on the ICON atmosphere processes.

    EBFM sends its surface and firn state (ice fraction, surface albedo, first firn layer
    temperature, conductance and heat capacity, runoff, surface mass balance, snow mass) to ICON-Land
    and receives the results of the surface energy balance computed by ICON-Land (JSBACH) on its
    glacier tile, averaged over the EBFM time step: surface temperature, melt and evapotranspiration.

    The results depend on the state, so the two are communicated in separate phases: surface_state,
    then (once ICON-Land has computed the surface energy balance from it) energy_balance. This lets
    the caller send the state, do other independent work (e.g. exchange with icon_atmo) while
    ICON-Land computes, and only then receive the results, instead of blocking on ICON-Land right
    after sending to it.
    """

    surface_state = ExchangeKeySet(
        name="surface state",
        source_keys={"icefract", "albedo", "t_sub", "ghf_cond", "hcap_sub", "runoff", "smb", "snowmass"},
    )
    energy_balance = ExchangeKeySet(
        name="surface energy balance",
        target_keys={"t_srf", "melt", "evapotrans"},
    )
    accepted_exchange_key_sets = (surface_state, energy_balance)

    def __init__(self, coupler: "Coupler", name: str = ComponentId.ICON_LAND.value):
        super().__init__(coupler, name)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to IconLand.
        """
        timestep = Timestep(value=time.time_step_iso8601())

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
                    name="runoff",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Runoff from the firn column (kg m-2 s-1)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="smb",
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
        if requested_key_set == self.surface_state:
            self._put_if_coupled("icefract", data_to_exchange)
            self._put_if_coupled("albedo", data_to_exchange)
            self._put_if_coupled("t_sub", data_to_exchange)
            self._put_if_coupled("ghf_cond", data_to_exchange)
            self._put_if_coupled("hcap_sub", data_to_exchange)
            self._put_if_coupled("runoff", data_to_exchange, transform=self._map_mass_flux_from_ebfm)
            self._put_if_coupled("smb", data_to_exchange, transform=self._map_mass_flux_from_ebfm)
            self._put_if_coupled("snowmass", data_to_exchange, transform=lambda x: x * 1e3)
            return {}

        # exchange() only calls _exchange for an accepted key set, so this is energy_balance.

        received_data: dict[str, np.ndarray] = {}

        t_srf = self._get_if_coupled("t_srf", fallback_values=fallback_values)
        if t_srf is not None:
            received_data["t_srf"] = t_srf

        for name in ("melt", "evapotrans"):
            data = self._get_if_coupled(name, transform=self._map_mass_flux_to_ebfm, fallback_values=fallback_values)
            if data is not None:
                received_data[name] = data

        return received_data
