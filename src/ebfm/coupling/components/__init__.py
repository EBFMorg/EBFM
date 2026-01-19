# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from .base import Component  # noqa: F401

# TODO: should not be necessary if ElmerIce etc. use a generic Field instead of (YAC)Field
from ebfm.coupling.couplers.helpers import coupling_supported

if coupling_supported:
    from .icon_atmo import IconAtmo  # noqa: F401
    from .elmer_ice import ElmerIce  # noqa: F401
