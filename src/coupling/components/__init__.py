# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from coupling.components.base import Component  # noqa: F401

# TODO: should not be necessary if ElmerIce etc. use a generic Field instead of (YAC)Field
from coupling.couplers.helpers import coupling_supported

if coupling_supported:
    from coupling.components.icon_atmo import IconAtmo  # noqa: F401
    from coupling.components.elmer_ice import ElmerIce  # noqa: F401
