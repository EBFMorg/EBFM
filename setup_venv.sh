#!/bin/bash

# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

set -e

if [[ -z "${YAC_BUILD_DIR}" ]]; then
  echo ERROR: Please export YAC_BUILD_DIR=/some/path/like/yac/build
  exit 1  # exit with error
fi

python -m venv .venv
source .venv/bin/activate
pip install $YAC_BUILD_DIR/python
cp $YAC_BUILD_DIR/python/yac/_yac.cpython-3*-x86_64-linux-gnu.so .venv/lib64/python3.*/site-packages/yac/_yac.cpython-3*-x86_64-linux-gnu.so 
pip install -r requirements.txt