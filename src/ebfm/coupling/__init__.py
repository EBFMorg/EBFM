# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

from .couplers import Coupler, DummyCoupler, FakeCoupler  # noqa: F401
from ebfm.core.config import CouplingLibrary

# The coupler of each coupling library: module, class, and how to install the library. A module is only imported when
# a run selects its library, so a run never loads a coupling library it does not use.
_COUPLERS: dict[CouplingLibrary, tuple[str, str, str]] = {
    CouplingLibrary.YAC: (
        ".couplers.yacCoupler",
        "YACCoupler",
        "install YAC and its python bindings as described under "
        "https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html",
    ),
}


def _import_coupler_class(library: CouplingLibrary) -> type[Coupler]:
    """Import the coupler of a coupling library, which imports the library itself.

    @param[in] library coupling library
    @returns Coupler subclass for the library
    @raises RuntimeError if the library or its python bindings cannot be imported
    """
    module_name, class_name, install_hint = _COUPLERS[library]
    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as e:
        raise RuntimeError(f"""
Coupling with {library.value} requires its python bindings, which could not be imported due to the following error:

{e}

Hint: {install_hint}
""") from e
    return getattr(module, class_name)


def check_coupling_requirements(coupling_config, active_coupling_features: list[str]) -> None:
    """Raise RuntimeError if real coupling is requested but the coupling library is not available.

    @param[in] coupling_config CouplingConfig holding coupling flags
    @param[in] active_coupling_features list of active coupling CLI flags (used in the error message)
    @raises RuntimeError if coupling is requested, fake mode is off, and the coupling library could not be imported
    """
    if coupling_config.use_fake_coupling:
        return  # Fake coupling bypasses requirements for real coupling backends

    if active_coupling_features:
        try:
            _import_coupler_class(CouplingLibrary.YAC)
        except RuntimeError as e:
            raise RuntimeError(f"Coupling requested via command line argument(s) {active_coupling_features}.{e}") from e


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
        return _import_coupler_class(CouplingLibrary.YAC)

    return DummyCoupler
