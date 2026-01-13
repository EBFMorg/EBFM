# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from coupling.couplers import yac_import_error
from coupling.couplers import Coupler, DummyCoupler  # noqa: F401

coupling_supported: bool
coupling_supported_import_error: Exception

if yac_import_error:
    coupling_supported = False
    coupling_supported_import_error = yac_import_error
else:
    coupling_supported = True
    coupling_supported_import_error = None

if coupling_supported:
    from coupling.couplers.yacCoupler import YACCoupler  # noqa: F401
    from coupling.couplers.oasisCoupler import OASISCoupler  # noqa: F401
