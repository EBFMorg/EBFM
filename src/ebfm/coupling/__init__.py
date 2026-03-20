# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from .couplers.helpers import coupling_supported
from .couplers.helpers import coupling_supported_import_error  # noqa: F401

from .couplers import Coupler, DummyCoupler, FakeCoupler  # noqa: F401

if coupling_supported:
    from .couplers.yacCoupler import YACCoupler  # noqa: F401
    from .couplers.oasisCoupler import OASISCoupler  # noqa: F401


def check_coupling_requirements(coupling_config, active_coupling_features: list[str]) -> None:
    """Raise RuntimeError if real coupling is requested but no backend is available.

    @param[in] coupling_config CouplingConfig holding coupling flags
    @param[in] active_coupling_features list of active coupling CLI flags (used in the error message)
    @raises RuntimeError if coupling is requested, fake mode is off, and YAC could not be imported
    """
    if coupling_config.use_fake_coupling:
        return  # Fake coupling bypasses requirements for real coupling backends

    coupling_used = len(active_coupling_features) > 0
    if coupling_used and not coupling_supported:
        raise RuntimeError(
            f"""
Coupling requested via command line argument(s) {active_coupling_features}, but the 'coupling' module could not be
imported due to the following error:

{coupling_supported_import_error}

Hint: If you are missing 'yac', please install YAC and the python bindings as described under
https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html"
"""
        )


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
