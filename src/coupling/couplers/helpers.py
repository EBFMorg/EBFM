# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

coupling_supported: bool
coupling_supported_import_error: Exception = None
yac_import_error: Exception = None

try:
    import yac  # noqa: F401
except ImportError as e:
    yac_import_error = e

if yac_import_error:
    coupling_supported = False
    coupling_supported_import_error = yac_import_error
else:
    coupling_supported = True
