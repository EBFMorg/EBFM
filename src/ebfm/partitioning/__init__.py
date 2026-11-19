# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre- and post-processing for runs on partitioned grids: split a restart file for a partitioned mesh, and merge the
per-partition output and restart files.

Each module runs standalone: python -m ebfm.partitioning.<module> --help.
"""
