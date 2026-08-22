# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING
from collections.abc import Mapping
import numpy as np

if TYPE_CHECKING:
    from ebfm.coupling.couplers.base import Coupler

from .base import Component

from ebfm.coupling.fields import FieldSet, Field, ExchangeType, Timestep
from ebfm.core.config import ComponentId, TimeConfig


class IconLand(Component):
    """
    Component class for coupling to the ICON land model (ICON-Land / JSBACH).

    The coupling to ICON-Land is separate from the coupling to the ICON atmosphere: ICON-Land
    is its own YAC component (``icon-land``) living on the ICON atmosphere processes.

    EBFM sends its surface state (ice fraction, surface albedo) to ICON-Land and receives the
    results of the surface energy balance computed by ICON-Land (JSBACH) on its glacier tile,
    averaged over the EBFM time step: surface temperature, melt, evapotranspiration and, for
    diagnostics, the surface energy fluxes.
    """

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
                Field(
                    name="hfss",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Sensible heat flux at the surface (W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="hfls",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Latent heat flux at the surface (W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="rsns",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Net surface shortwave radiation (W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="rlns",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Net surface longwave radiation (W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
                Field(
                    name="ghf",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Ground heat flux (W m-2)",
                    exchange_type=ExchangeType.TARGET,
                ),
            }
        )

    def exchange(
        self, data_to_exchange: Mapping[str, np.ndarray], fallback_values: Mapping[str, np.ndarray] = {}
    ) -> dict[str, np.ndarray]:
        """
        Exchange data with IconLand.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values optional Mapping of field names to fallback values to use if get fails

        @returns dictionary of received field data: "t_srf" (K), "melt" and "evapotrans"
                 (m w.e. per EBFM time step, evapotrans negative upward), "hfss", "hfls",
                 "rsns", "rlns", "ghf" (W m-2); only the fields that are actually coupled
        """
        received_data: dict[str, np.ndarray] = {}

        # Put data to IconLand
        self._put_if_coupled("icefract", data_to_exchange)
        self._put_if_coupled("albedo", data_to_exchange)

        # Get data from IconLand
        for name in ("t_srf", "hfss", "hfls", "rsns", "rlns", "ghf"):
            data = self._get_if_coupled(name, fallback_values=fallback_values)
            if data is not None:
                received_data[name] = data

        for name in ("melt", "evapotrans"):
            data = self._get_if_coupled(name, transform=self._map_mass_flux_to_ebfm, fallback_values=fallback_values)
            if data is not None:
                received_data[name] = data

        return received_data
