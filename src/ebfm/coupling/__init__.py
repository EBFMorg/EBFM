# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from .couplers.helpers import coupling_supported
from .couplers.helpers import coupling_supported_import_error  # noqa: F401

from .couplers import Coupler, DummyCoupler, FakeCoupler  # noqa: F401

if coupling_supported:
    from .couplers.yacCoupler import YACCoupler  # noqa: F401
    from .couplers.oasisCoupler import OASISCoupler  # noqa: F401


def select_coupler_class(coupling_config) -> type[Coupler]:
    """Factory pattern: return the Coupler subclass for this run.

    - No coupling requested          → DummyCoupler (no-op)
    - Coupling requested, fake mode  → FakeCoupler (no YAC required)
    - Coupling requested, real mode  → YACCoupler

    @param[in] coupling_config CouplingConfig holding coupling flags
    @returns Coupler subclass to instantiate
    @raises RuntimeError if real coupling is requested but YAC is not available
    """
    if coupling_config.defines_coupling():
        if coupling_config.use_fake_coupling:
            return FakeCoupler
        if not coupling_supported:
            raise RuntimeError("Coupling is requested but no supported coupling backend is available.")
        return YACCoupler

    return DummyCoupler
