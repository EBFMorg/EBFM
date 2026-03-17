# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Dict as TypingDict, List, Optional, Tuple

import numpy as np

from ebfm.core import logging
from ebfm.core.config import CouplingConfig

from .base import Coupler, CouplerErrorCode, Grid, Dict

logger = logging.getLogger(__name__)


@dataclass
class FakeFieldConfig:
    """
    Configuration for a single fake field returned by FakeCoupler.

    The scalar ``value`` is broadcast to all grid points when the field
    is requested via :meth:`FakeCoupler.get`.
    """

    component_name: str
    """Name of the coupled component this field belongs to (e.g. ``"elmer_ice"``)."""

    field_name: str
    """Name of the field (e.g. ``"h"``)."""

    value: float = 0.0
    """Scalar fill value used to construct the returned array."""


# ---------------------------------------------------------------------------
# Default fake values for all known TARGET fields (fields that EBFM receives).
# SOURCE fields sent by EBFM (T_ice, smb, runoff …) are silently discarded by
# put() and therefore need no entry here.
# ---------------------------------------------------------------------------
_DEFAULT_FAKE_FIELDS: List[FakeFieldConfig] = [
    # Elmer/Ice → EBFM
    FakeFieldConfig("elmer_ice", "h", 1000.0),  # surface height            [m]
    # ICON atmosphere → EBFM
    FakeFieldConfig("icon_atmo", "pr", 0.0),  # precipitation rate        [kg m-2 s-1]
    FakeFieldConfig("icon_atmo", "pr_snow", 0.0),  # snowfall rate             [kg m-2 s-1]
    FakeFieldConfig("icon_atmo", "rsds", 100.0),  # downward SW radiation     [W m-2]
    FakeFieldConfig("icon_atmo", "rlds", 300.0),  # downward LW radiation     [W m-2]
    FakeFieldConfig("icon_atmo", "sfcwind", 5.0),  # surface wind speed        [m s-1]
    FakeFieldConfig("icon_atmo", "clt", 0.5),  # cloud cover               [fraction]
    FakeFieldConfig("icon_atmo", "tas", 260.0),  # near-surface temperature  [K]
    FakeFieldConfig("icon_atmo", "huss", 1e-3),  # specific humidity         [kg kg-1]
    FakeFieldConfig("icon_atmo", "sfcpres", 101325.0),  # surface pressure          [Pa]
]


class FakeCoupler(Coupler):
    """
    A fake coupler that returns synthetic data instead of real coupled data.

    This coupler is useful for running EBFM standalone without needing the actual
    coupling library (YAC) or coupled models (ICON atmosphere, Elmer/Ice) to be
    running. Unlike DummyCoupler which stands for no active coupling, FakeCoupler
    is intended for testing the coupling infrastructure by providing fake values
    to emulate the presence of coupled models.

    Fake field values are supplied at construction time via a list of
    :class:`FakeFieldConfig` objects.  If none are given, :data:`_DEFAULT_FAKE_FIELDS`
    is used, which provides physically plausible constants for all known TARGET fields.

    Usage example::

        # Use built-in defaults
        coupler = FakeCoupler(coupling_config)

        # Or override specific fields
        coupler = FakeCoupler(coupling_config, fake_fields=[
            FakeFieldConfig("elmer_ice", "h", 500.0),
            FakeFieldConfig("icon_atmo", "tas", 270.0),
        ])

        coupler.setup(grid, time)
        data, err = coupler.get("elmer_ice", "h")   # returns np.full(n_points, 500.0)
        coupler.put("elmer_ice", "smb", smb_data)   # silently discarded
    """

    def __init__(self, coupling_config: CouplingConfig, fake_fields: Optional[List[FakeFieldConfig]] = None):
        """
        Create a FakeCoupler and pre-load fake field values.

        Does not require YAC or any coupled model to be available.

        @param[in] coupling_config coupling configuration of this component
        @param[in] fake_fields     list of :class:`FakeFieldConfig` entries that define
                                   which fields are available and what scalar value they
                                   return.  Defaults to :data:`_DEFAULT_FAKE_FIELDS`.
        """
        super().__init__(coupling_config)

        self.field_validation_level = coupling_config.field_validation_level

        # Scalar fill values keyed by (component_name, field_name).
        # Arrays are constructed lazily in get() once _n_points is known.
        self._fake_values: TypingDict[Tuple[str, str], float] = {}
        self._n_points: int = 0

        fields_to_load = fake_fields if fake_fields is not None else _DEFAULT_FAKE_FIELDS
        for cfg in fields_to_load:
            self._fake_values[(cfg.component_name, cfg.field_name)] = cfg.value

        logger.debug(
            f"FakeCoupler created for component '{coupling_config.component_name}' "
            f"with {len(self._fake_values)} fake field(s)."
        )

    # TODO: Try to improve this
    def _infer_n_points(self, grid: Optional[Dict | Grid]) -> int:
        """
        Infer the number of horizontal points represented by ``grid``.

        Supports both:
        - Elmer mesh-like objects (e.g. ``vertex_ids``, ``lon``, ``lat``)
        - MATLAB/full EBFM grid dictionaries (e.g. ``x``, ``lon``, ``lat``, ``gpsum``)
        """
        if grid is None:
            return 0

        if isinstance(grid, dict):
            for key in ("n_points", "x", "lon", "lat", "mask"):
                value = grid.get(key)
                if value is not None:
                    try:
                        return int(len(value))
                    except TypeError:
                        pass

            gpsum = grid.get("gpsum")
            if gpsum is not None:
                return int(gpsum)

            return 0

        for attr in ("vertex_ids", "lon", "lat", "x_vertices", "y_vertices"):
            value = getattr(grid, attr, None)
            if value is not None:
                try:
                    return int(len(value))
                except TypeError:
                    pass

        return 0

    def setup(self, grid: Dict | Grid, time: TypingDict[str, float]):
        """
        Store grid size so fake arrays can be sized correctly in :meth:`get`.

        No synchronization with external models is performed.

        @param[in] grid Grid used by EBFM where coupling happens
        @param[in] time dictionary with time parameters, e.g. {'tn': 12, 'dt': 0.125}
        """
        self._n_points = self._infer_n_points(grid)
        logger.debug(f"FakeCoupler setup complete ({self._n_points} grid points, no sync performed).")

    def put(self, component_name: str, field_name: str, data: np.ndarray) -> Optional[CouplerErrorCode]:
        """
        Log and discard outgoing data – no actual transfer is performed.

        @param[in] component_name name of the component to put data to
        @param[in] field_name name of the field to put data to
        @param[in] data data that would be sent to the coupled model

        @returns None (no error)
        """
        logger.debug(f"FakeCoupler put: field '{field_name}' -> '{component_name}' (discarded).")
        return None

    def get(self, component_name: str, field_name: str) -> Tuple[Optional[np.ndarray], Optional[CouplerErrorCode]]:
        """
        Return a fake array for the requested field.

        The array is constructed by broadcasting the scalar configured for
        ``(component_name, field_name)`` across all grid points.  If no entry
        exists for the requested pair, an array of zeros is returned.

        @param[in] component_name name of the component to get data from
        @param[in] field_name name of the field to retrieve

        @returns tuple of (fake field data, error code). Error code is always None.
        """
        key = (component_name, field_name)
        fill_value = self._fake_values.get(key, 0.0)
        if key not in self._fake_values:
            logger.debug(
                f"FakeCoupler get: no fake value configured for '{field_name}' from "
                f"'{component_name}', returning zeros."
            )
        else:
            logger.debug(
                f"FakeCoupler get: returning np.full({self._n_points}, {fill_value}) "
                f"for '{field_name}' from '{component_name}'."
            )
        return np.full(self._n_points, fill_value), None

    def finalize(self):
        """
        Finalize the coupling interface (no-op for FakeCoupler).
        """
        logger.info("FakeCoupler finalized.")
