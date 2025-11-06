# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import argparse

import numpy as np
from numpy.typing import NDArray

from elmer.mesh import TriangleMesh
import elmer.parser


def read_elmer_mesh(mesh_root: Path, is_partitioned: bool = False, partition_id: int = -1) -> TriangleMesh:
    """Read Elmer mesh files.
    Args:
        mesh_root (Path): Path to the Elmer mesh folder.
        is_partitioned (bool): Set True if given mesh is partitioned.
        partition_id (int): Provide partition_id if is_partitioned=True. Identifies partition.
    Returns:
        Mesh: A Mesh object containing x, y, z coordinates, vertex IDs, cell-to-vertex mapping, and cell IDs.
    """
    # Check input
    assert mesh_root.is_dir(), f"{mesh_root} is no directory or does not exist."

    if not is_partitioned:
        # Check subdirectories of mesh_root
        header_file: Path = mesh_root / "mesh.header"
        nodes_file: Path = mesh_root / "mesh.nodes"
        elements_file: Path = mesh_root / "mesh.elements"
    else:
        # Check subdirectories of mesh_root
        header_file: Path = mesh_root / f"part.{partition_id}.header"
        nodes_file: Path = mesh_root / f"part.{partition_id}.nodes"
        elements_file: Path = mesh_root / f"part.{partition_id}.elements"

    assert (
        header_file.is_file()
    ), f"Header file {header_file} does not exist. Please ensure that this file exists in {mesh_root}."
    assert (
        nodes_file.is_file()
    ), f"Nodes file {nodes_file} does not exist. Please ensure that this file exists in {mesh_root}."
    assert (
        elements_file.is_file()
    ), f"Mesh file {elements_file} does not exist. Please ensure that this file exists in {mesh_root}."

    # Parse header, nodes, and elements files
    n_vertices, n_cells = elmer.parser.parse_header(header_file)
    global_vertex_ids, x_vertices, y_vertices, z_vertices = elmer.parser.parse_nodes(nodes_file)
    local_vertex_ids = range(len(global_vertex_ids))  # use [0,1,...,n_vertices-1] to identify vertices locally

    assert (
        len(global_vertex_ids) == n_vertices
    ), f"Number of vertex IDs in nodes file ({len(global_vertex_ids)}) does not match the header ({n_vertices})."
    assert (
        len(x_vertices) == n_vertices
    ), f"Number of vertices in nodes file ({len(x_vertices)}) does not match the header ({n_vertices})."
    assert (
        len(y_vertices) == n_vertices
    ), f"Number of vertices in nodes file ({len(y_vertices)}) does not match the header ({n_vertices})."
    assert (
        len(z_vertices) == n_vertices
    ), f"Number of vertices in nodes file ({len(z_vertices)}) does not match the header ({n_vertices})."

    global_cell_ids, global_cell_to_vertex = elmer.parser.parse_elements(elements_file)

    vertex_l2g = {
        loc: glob for loc, glob in zip(local_vertex_ids, global_vertex_ids)
    }  # map local ids [0,1,...n_verts] to global_vertex_ids
    vertex_g2l = {
        glob: loc for loc, glob in vertex_l2g.items()
    }  # invert dictionary to get map for global ids to local ids

    cell_to_vertex_local = np.array([[vertex_g2l[g_v] for g_v in c] for c in global_cell_to_vertex])

    assert (
        len(global_cell_ids) == n_cells
    ), f"Number of cell IDs in elements file ({len(global_cell_ids)}) does not match the header ({n_cells})."

    return TriangleMesh(
        x_vertices=x_vertices,
        y_vertices=y_vertices,
        z_vertices=z_vertices,
        cell_to_vertex=cell_to_vertex_local,
        vertex_ids=global_vertex_ids,
        cell_ids=global_cell_ids,
    )


def read_dem(dem_file: Path, xs: NDArray[np.float64], ys: NDArray[np.float64]):
    """Read digital elevation model (DEM) file.
    Args:
        dem_file (Path): Path to the DEM NetCDF file.
        xs (NDArray[np.float64]): x-coordinates to sample.
        ys (NDArray[np.float64]): y-coordinates to sample.
    Returns:
        NDArray[np.float64]: A 1D array of sampled heights at the given x and y coordinates.
    """
    assert dem_file.is_file(), f"DEM file {dem_file} does not exist."

    import netCDF4

    nc = netCDF4.Dataset(dem_file)
    # find all matching ids in x direction
    idx_x = np.searchsorted(nc["x"], xs)

    # find all matching ids in y direction
    # nc['y'] is in reversed order. So finding the matches is a bit more complicated since np.searchsorted expects and
    # array in ascending order sorter=reversed(...) allows us to find indices in an array with descending order
    idx_y = np.searchsorted(nc["y"], ys, sorter=list(reversed(range(len(nc["y"])))))
    idx_y *= -1  # we have invert the sign since indices of the reversed array are counted from the back
    idx_y += -1

    print(
        f"{np.sum(abs(xs - nc['x'][idx_x][:]) > 10e-5)} of {len(idx_x)} x-coordinates have a significant mismatch."
        " Surface value of a neighboring point will be used."
    )
    print(
        f"{np.sum(abs(ys - nc['y'][idx_y][:]) > 10e-5)} of {len(idx_y)} y-coordinates have a significant mismatch."
        " Surface value of a neighboring point will be used."
    )

    surf = nc["surface"][:]  # get numpy array
    return surf[idx_y, idx_x]  # sample at given indexes and return


def read_matlab(
    mat_file: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read custom grid from a MATLAB file.
    Args:
        mat_file (Path): Path to the MATLAB file.
    Returns:
        tuple: x, y coordinates and height data.
    """
    raise Exception("Reading from MATLAB files is not implemented yet.")


def write_dem_as_elmer(
    mesh: TriangleMesh,
    h: NDArray[np.float64],
    dem_file: Path,
    allow_overwrite: bool = False,
) -> None:
    """Write digital elevation model to a file following the structure of an existing Elmer mesh
    Args:
        mesh (Mesh): The mesh object containing x and y vertices and vertex IDs.
        h (NDArray[np.float64]): Height data to write.
        dem_file (Path): Path to the output DEM file.
    """

    if not allow_overwrite:
        assert not dem_file.is_file(), f"DEM file {dem_file} already exists. Please choose a different file name."

    assert len(h) == len(
        mesh.x_vertices
    ), f"Height data length ({len(h)}) does not match number of vertices ({len(mesh.x_vertices)})."

    import pandas as pd

    # Create a DataFrame with the required structure
    df = pd.DataFrame(
        {
            "Node ID": mesh.vertex_ids,
            "Node Type": -1,
            "x": mesh.x_vertices,
            "y": mesh.y_vertices,
            "z": h,
        }
    )

    def fortran_style_sci(x, precision=15):
        """Convert a number to Fortran-style scientific notation."""
        if x == 0:
            # special case for zero; Fortran style requires leading space for positive numbers
            return f" 0.{''.join(['0'] * precision)}E+00"

        exp = int(np.floor(np.log10(abs(x)))) + 1
        mantissa = x / (10**exp)
        sign = "-" if mantissa < 0 else " "
        return sign + f"{abs(mantissa):.{precision}f}E{exp:+03d}"

    import csv

    df.to_csv(
        dem_file,
        sep=" ",
        float_format=fortran_style_sci,
        index=False,
        header=False,
        escapechar="\\",  # required when using QUOTE_NONE with special chars
        quoting=csv.QUOTE_NONE,  # do not use quotes around fields
    )

    # Postprocess file to ensure it matches Elmer's expected format
    with open(dem_file, "r") as f:
        content = (
            "\n".join(line.rstrip() + " " for line in f.read().splitlines())  # append space to each line
            .replace("\\ ", " ")  # replace escaped spaces with actual spaces
            .replace(" -1 ", " -1  ")  # add another space after node type
        )

    with open(dem_file, "w") as f:
        f.write(content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read Elmer mesh and DEM files.")
    parser.add_argument("elmer_mesh", type=Path, help="Path to the Elmer mesh file.")
    parser.add_argument("dem", type=Path, help="Path to the digital elevation model (DEM) NetCDF file.")
    args = parser.parse_args()

    print("I'm running as main...")
    print(f"Reading the following files: {args.elmer_mesh} and {args.dem}")

    mesh = read_elmer_mesh(args.elmer_mesh)
    x = mesh.x_vertices
    y = mesh.y_vertices
    h = read_dem(args.dem, x, y)

    write_dem_as_elmer(mesh, h, args.elmer_mesh / "MESH" / "mesh.nodes", allow_overwrite=True)
