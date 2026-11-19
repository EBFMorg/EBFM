# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split a structured grid, a MATLAB or NetCDF grid file, into partitions that EBFM can run on, one MPI rank each.

Writes <grid>.partitioning.<n>/part.<i>.<suffix> next to the grid file <grid>.<suffix>, for i = 1..n. Partition i
holds the i-th of n contiguous blocks of the active cells, in the order of the file. Each partition is a copy of the
grid file whose mask switches off the active cells of all other partitions, so EBFM reads it like the whole grid, with
the same 2-D fields and the same column ids.

On n MPI ranks, EBFM runs on <grid>.partitioning.<n> when given --matlab-mesh or --netcdf-mesh <grid>.<suffix>, and
creates it with this module if it is missing and --create-partitioning is given. To create it beforehand:

    python -m ebfm.grid.structured.partitioning examples/dem_and_mask.mat 4
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from netCDF4 import Dataset

from ebfm.grid.partitioning import write_directory_atomically
from ebfm.grid.structured.matlab import read_MATLAB_fields
from ebfm.grid.structured.netcdf import read_NETCDF_grid

# The mask value of an active cell in a MATLAB grid file
_MATLAB_ACTIVE = 1


def partitioning_dir(grid_file: Path, n_partitions: int) -> Path:
    """The directory of the partitioning of grid_file into n_partitions."""
    grid_file = Path(grid_file)
    return grid_file.with_name(f"{grid_file.stem}.partitioning.{n_partitions}")


def partition_file(partitioning: Path, partition_id: int, grid_file: Path) -> Path:
    """The file of partition partition_id in the partitioning of grid_file."""
    return Path(partitioning) / f"part.{partition_id}{Path(grid_file).suffix}"


def partition_column_ids(partitioning: Path) -> dict[int, np.ndarray]:
    """The column ids of every partition of a partitioning of a structured grid, by partition id, in id order."""
    files = {}
    for suffix in (".mat", ".nc"):
        files.update({int(f.name.split(".")[1]): f for f in Path(partitioning).glob(f"part.*{suffix}")})
    if sorted(files) != list(range(1, len(files) + 1)):
        raise ValueError(f"{partitioning} holds no partitions part.<i>.mat or part.<i>.nc numbered from 1.")
    return {partition_id: column_ids(files[partition_id]) for partition_id in sorted(files)}


def column_ids(grid_file: Path) -> np.ndarray:
    """The column ids of a MATLAB or NetCDF grid file, as EBFM reads them: the flat indices of its active cells."""
    if Path(grid_file).suffix == ".mat":
        mask = read_MATLAB_fields(grid_file)["mask"]
    else:
        mask = read_NETCDF_grid(grid_file)["mask"]
    return np.flatnonzero(mask == 1)


def partition_grid(grid_file: Path, n_partitions: int, out_dir: Path | None = None) -> Path:
    """
    Write n_partitions partitions of grid_file to out_dir, partitioning_dir() by default.

    The partitions are written atomically (see ebfm.grid.partitioning.write_directory_atomically).

    @returns out_dir
    @raises FileExistsError if out_dir exists
    @raises ValueError if grid_file is neither a MATLAB (.mat) nor a NetCDF (.nc) file, or has too few active cells
    """
    grid_file = Path(grid_file)
    writers = {".mat": _write_matlab_partitions, ".nc": _write_netcdf_partitions}
    if grid_file.suffix not in writers:
        raise ValueError(f"{grid_file} is neither a MATLAB (.mat) nor a NetCDF (.nc) grid file.")
    if n_partitions < 1:
        raise ValueError(f"The number of partitions must be at least 1, not {n_partitions}.")

    out_dir = Path(out_dir) if out_dir is not None else partitioning_dir(grid_file, n_partitions)
    write = writers[grid_file.suffix]
    return write_directory_atomically(out_dir, lambda tmp_dir: write(grid_file, n_partitions, tmp_dir))


def _blocks(active: np.ndarray, n_partitions: int, grid_file: Path) -> list[np.ndarray]:
    """The flat indices of the active cells of each partition."""
    if len(active) < n_partitions:
        raise ValueError(f"{grid_file} has {len(active)} active cells, too few for {n_partitions} partitions.")
    return np.array_split(active, n_partitions)


def _write_matlab_partitions(grid_file: Path, n_partitions: int, out_dir: Path) -> None:
    grid = sio.loadmat(str(grid_file))["grid_svalbard"]
    mask = grid["mask"][0, 0]
    active = np.flatnonzero(mask == _MATLAB_ACTIVE)

    for partition_id, block in enumerate(_blocks(active, n_partitions, grid_file), start=1):
        part_mask = mask.copy()
        part_mask.flat[np.setdiff1d(active, block)] = 0
        part = grid.copy()
        part["mask"][0, 0] = part_mask
        sio.savemat(str(partition_file(out_dir, partition_id, grid_file)), {"grid_svalbard": part})


def _write_netcdf_partitions(grid_file: Path, n_partitions: int, out_dir: Path) -> None:
    # The active cells as EBFM reads them, which takes the orography into account
    active = np.flatnonzero(read_NETCDF_grid(grid_file)["mask"] == 1)

    for partition_id, block in enumerate(_blocks(active, n_partitions, grid_file), start=1):
        part = partition_file(out_dir, partition_id, grid_file)
        shutil.copyfile(grid_file, part)
        with Dataset(part, "r+") as nc:
            mask = nc.variables["mask"]
            values = mask[:]
            # The mask may carry dimensions of size 1, which the reader squeezes out
            flat = values.reshape(-1)
            flat[np.setdiff1d(active, block)] = 0
            mask[:] = flat.reshape(values.shape)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid_file", type=Path, help="MATLAB (.mat) or NetCDF (.nc) grid file")
    parser.add_argument("n_partitions", type=int, help="number of partitions, i.e. of MPI ranks to run EBFM on")
    parser.add_argument(
        "-o", "--output", type=Path, help="directory to write the partitions to (default: <grid>.partitioning.<n>)"
    )
    args = parser.parse_args(argv)

    try:
        out_dir = partition_grid(args.grid_file, args.n_partitions, args.output)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {args.n_partitions} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
