# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre- and post-processing of EBFM's output and restart files for runs on partitioned grids: merge the files of the
partitions into one, and split a restart file into one per partition. Each module runs standalone:
python -m ebfm.io.<module> --help.
"""
