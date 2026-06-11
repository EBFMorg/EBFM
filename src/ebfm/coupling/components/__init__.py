# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from ...core.config import ComponentId

from .base import Component  # noqa: F401
from .icon_atmo import IconAtmo
from .elmer_ice import ElmerIce

# maps a given ID to the component implementation to be used
id2class = {
    ComponentId.ELMER_ICE: ElmerIce,
    ComponentId.ICON_ATMO: IconAtmo,
}
