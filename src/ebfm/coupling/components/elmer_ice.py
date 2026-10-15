# SPDX-FileCopyrightText: 2025 EBFM Authors
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
from ebfm.core.constants import DAYS_PER_YEAR
from ebfm.core.grid import GridDict


class ElmerIce(Component):
    """
    Component class for Elmer/Ice model coupling.
    """

    accepted_exchange_key_sets = (
        # All data is exchanged at once, i.e. the caller has to send and receive everything in a single call.
        ExchangeKeySet(
            name="exchange",
            source_keys={"T_ice", "smb_to_elmer", "runoff_to_elmer"},
            target_keys={
                "surface_elevation",
                # Enable together with their field definitions and their gets in _exchange below.
                # "dhdx",
                # "dhdy",
            },
        ),
    )

    def __init__(self, coupler: "Coupler", name: str = ComponentId.ELMER_ICE.value):
        super().__init__(coupler, name)

    def get_field_definitions(self) -> FieldSet:
        """
        Get generic field definitions for EBFM coupling to Elmer/Ice.
        """
        timestep = Timestep(value=self.ebfm_time.time_step_iso8601())

        return FieldSet(
            {
                Field(
                    name="T_ice",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Near surface temperature at Ice surface (in K)",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="smb_to_elmer",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface mass balance",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="runoff_to_elmer",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Runoff",
                    exchange_type=ExchangeType.SOURCE,
                ),
                Field(
                    name="surface_elevation",
                    coupled_component=self,
                    timestep=timestep,
                    metadata="Surface elevation (in m)",
                    exchange_type=ExchangeType.TARGET,
                ),
                # Field(
                #     name="dhdx",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Surface slope in x direction",
                #     exchange_type=ExchangeType.TARGET,
                # ),
                # Field(
                #     name="dhdy",
                #     coupled_component=self,
                #     timestep=timestep,
                #     metadata="Surface slope in y direction",
                #     exchange_type=ExchangeType.TARGET,
                # ),
            }
        )

    def validate_grid(self, grid: GridDict):
        """
        Check that the grid tolerates a surface elevation that changes over time.

        update_surface_elevation lets grid["z"] follow the ice surface, while everything INIT derives from the
        elevation keeps describing the geometry read during initialization. These are the conditions under
        which that difference does not matter. One case stays unguarded: grid["mesh"] keeps its initial vertex
        and cell elevations, which the mesh topology in the output file is written from.

        @param[in] grid grid EBFM runs on

        @raises AssertionError if a quantity derived from the initial elevation is in use and would go stale
        """
        # Both shading methods rest on horizon angles that INIT computes once from the initial elevation.
        assert not grid["has_shading"], "Shading does not support a surface elevation that changes over time."

        # The gradient fields are not exchanged yet (see _exchange), so the slopes cannot follow the surface.
        # Zero slopes carry no geometry, which is why an elevation update is admissible at all.
        for slope_field in ("slope_x", "slope_y", "slope_beta", "slope_gamma"):
            assert np.all(
                grid[slope_field] == 0.0
            ), f"Grid field '{slope_field}' is non-zero, but slopes are not updated from '{self.name}'."

        # A MATLAB grid holds the elevation a second time, on the 2-D grid that structured output is written from.
        assert "z_2D" not in grid, f"A 2-D elevation field is not updated from '{self.name}'."

    def update_surface_elevation(self, grid: GridDict, surface_elevation: np.ndarray):
        """
        Let the EBFM grid follow the ice surface that Elmer/Ice reports back.

        Only the elevation itself follows. The conditions under which that is admissible are checked once
        during setup, see validate_grid.

        @param[in,out] grid grid whose elevation is updated in place
        @param[in] surface_elevation elevation received from Elmer/Ice, one value per column

        @raises AssertionError if the received elevation does not cover the grid
        """
        assert surface_elevation.shape == grid["z"].shape, (
            f"Component '{self.name}' reported a surface elevation of shape {surface_elevation.shape}, "
            f"expected one value per column, i.e. shape {grid['z'].shape}."
        )

        grid["z"] = surface_elevation

    def _exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray],
        requested_key_set: ExchangeKeySet,
    ) -> dict[str, np.ndarray]:
        """
        Exchange data with Elmer/Ice.

        This component accepts a single key set, so everything is sent and received here.

        @param[in] data_to_exchange read-only Mapping of field names to data to be sent
        @param[in] fallback_values Mapping of field names to fallback values to use if get fails
        @param[in] requested_key_set key set to be communicated, the only one this component accepts

        @returns dictionary of received field data. A requested field is not contained if it is not coupled, or
                 if no data was received for it and no fallback value was given.
        """
        received_data: dict[str, np.ndarray] = {}

        # For fields representing rates (e.g. SMB, runoff), we need to convert them from per timestep to per year
        # before sending to Elmer/Ice, which expects annual values.
        def map_per_timestep_to_per_year(x_per_timestep: np.ndarray) -> np.ndarray:
            x_per_day = x_per_timestep / self.ebfm_time.time_step_in_days()
            x_per_year = x_per_day * DAYS_PER_YEAR
            return x_per_year

        # Put data to Elmer/Ice
        self._put_if_coupled("T_ice", data_to_exchange)
        self._put_if_coupled("smb_to_elmer", data_to_exchange, transform=map_per_timestep_to_per_year)
        self._put_if_coupled("runoff_to_elmer", data_to_exchange, transform=map_per_timestep_to_per_year)

        # Get data from Elmer/Ice
        surface_elevation = self._get_if_coupled("surface_elevation", fallback_values=fallback_values)
        if surface_elevation is not None:
            received_data["surface_elevation"] = surface_elevation

        # The gradient fields have no field definition yet, so these two gets do nothing. Enabling their field
        # definitions also requires uncommenting them in the accepted key set above, because the target keys a
        # caller requests by default are the coupled fields of this component.
        dhdx = self._get_if_coupled("dhdx", fallback_values=fallback_values)
        if dhdx is not None:
            received_data["dhdx"] = dhdx

        dhdy = self._get_if_coupled("dhdy", fallback_values=fallback_values)
        if dhdy is not None:
            received_data["dhdy"] = dhdy

        return received_data
