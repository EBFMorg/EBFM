# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""What the partitionings of all kinds of grids share."""

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path


def write_directory_atomically(out_dir: Path, write: Callable[[Path], None]) -> Path:
    """
    Create out_dir with the files that write puts into an empty directory.

    write fills a temporary directory next to out_dir, which then takes its place, so that out_dir never holds a
    partial partitioning, e.g. after an interrupted run or while another run writes the same one.

    @returns out_dir
    @raises FileExistsError if out_dir exists
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists.")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        write(tmp_dir)
        os.rename(tmp_dir, out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return out_dir
