# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre- and post-processing for runs on partitioned grids: partition an Elmer mesh, split a restart file for it, and
merge the per-partition output and restart files.

EBFM uses them itself, e.g. to create a missing partitioning with --create-partitioning, and each module also runs
standalone: python -m ebfm.partitioning.<module> --help.
"""
