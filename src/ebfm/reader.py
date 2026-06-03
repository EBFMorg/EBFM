# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import shutil

import argparse

import numpy as np
from numpy.typing import NDArray

from ebfm.elmer.mesh import TriangleMesh
import ebfm.elmer.parser

import logging

logger = logging.getLogger(__name__)


def read_elmer_mesh(
    mesh_root: Path,
    is_partitioned: bool = False,
    partition_id: int = -1,
    source_crs_epsg: int = 3413,
) -> TriangleMesh:
    """Read Elmer mesh files.
    Args:
        mesh_root (Path): Path to the Elmer mesh folder.
        is_partitioned (bool): Set True if given mesh is partitioned.
        partition_id (int): Provide partition_id if is_partitioned=True. Identifies partition.
        source_crs_epsg (int): EPSG code of the Elmer mesh x/y coordinates.
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
    n_vertices, n_cells = ebfm.elmer.parser.parse_header(header_file)
    global_vertex_ids, x_vertices, y_vertices, z_vertices = ebfm.elmer.parser.parse_nodes(nodes_file)
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

    global_cell_ids, global_cell_to_vertex = ebfm.elmer.parser.parse_elements(elements_file)

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
        source_crs_epsg=source_crs_epsg,
    )


def read_dem_xios(dem_file: Path, mesh: TriangleMesh):
    """Read digital elevation model (DEM) file.

    Reads vertex-level surface elevation (zs) and ice thickness (h) from NetCDF
    and maps to cell centers via arithmetic averaging of scalar field values.
    (For coordinate averaging, see mesh._compute_cell_centers() which uses spherical geometry.)

    Args:
        dem_file (Path): Path to the DEM NetCDF file.
        mesh (TriangleMesh): Mesh object with vertex and cell information
    Returns:
        tuple: (z, h) arrays with cell-center values in meters
    """
    assert dem_file.is_file(), f"DEM file {dem_file} does not exist."

    import netCDF4

    nc = netCDF4.Dataset(dem_file)

    # Read vertex data from nc file
    zs_vertices = np.squeeze(nc["zs"][:]).data
    h_vertices = np.squeeze(nc["h"][:]).data

    assert len(zs_vertices) == len(mesh.x_vertices), (
        f"Surface mesh ({len(zs_vertices)} vertices) and Elmer mesh "
        f"({len(mesh.x_vertices)} vertices) do not have the same number of vertices"
    )

    # Map vertex values to cell centers by arithmetic averaging (scalar fields)
    n_cells = mesh.cell_to_vertex.shape[0]

    z_cells = np.zeros(n_cells)
    h_cells = np.zeros(n_cells)

    for i in range(n_cells):
        vertex_indices = mesh.cell_to_vertex[i]
        z_cells[i] = np.mean(zs_vertices[vertex_indices])
        h_cells[i] = np.mean(h_vertices[vertex_indices])

    return z_cells, h_cells


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

    def nearest_indices(axis_values: NDArray[np.float64], query_values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return nearest-neighbor indices for query values on a monotonic axis."""
        assert axis_values.size > 0, "Axis array is empty."

        diffs = np.diff(axis_values)
        is_ascending = np.all(diffs > 0)
        is_descending = np.all(diffs < 0)
        is_monotonic = is_ascending or is_descending

        if not is_monotonic:
            raise ValueError("Axis values must be strictly monotonic (strictly increasing or strictly decreasing).")

        axis_for_search = axis_values if is_ascending else axis_values[::-1]

        insert_pos = np.searchsorted(axis_for_search, query_values)
        insert_pos = np.clip(insert_pos, 0, len(axis_for_search) - 1)

        left_pos = np.clip(insert_pos - 1, 0, len(axis_for_search) - 1)
        right_pos = insert_pos
        choose_right = np.abs(axis_for_search[right_pos] - query_values) <= np.abs(
            axis_for_search[left_pos] - query_values
        )
        nearest_pos = np.where(choose_right, right_pos, left_pos)

        if not is_ascending:
            nearest_pos = len(axis_values) - 1 - nearest_pos

        return nearest_pos.astype(np.int64)

    with netCDF4.Dataset(dem_file) as nc:
        x_axis = np.asarray(nc["x"][:], dtype=np.float64)
        y_axis = np.asarray(nc["y"][:], dtype=np.float64)

        idx_x = nearest_indices(x_axis, xs)
        idx_y = nearest_indices(y_axis, ys)

        # Normalize absolute coordinate mismatch by typical axis spacing.
        # This keeps diagnostics meaningful near x/y ~= 0.
        x_spacing = np.median(np.abs(np.diff(x_axis)))
        y_spacing = np.median(np.abs(np.diff(y_axis)))
        x_spacing = max(x_spacing, np.finfo(np.float64).eps)
        y_spacing = max(y_spacing, np.finfo(np.float64).eps)

        normalized_error_tol = 0.5  # warn if nearest neighbor is more than 0.5 typical spacings away from query point
        mismatch_x = np.abs(xs - x_axis[idx_x]) / x_spacing
        mismatch_y = np.abs(ys - y_axis[idx_y]) / y_spacing

        if np.any(mismatch_x > normalized_error_tol) or np.any(mismatch_y > normalized_error_tol):

            def print_axis_diagnostics(
                axis_name: str,
                axis_values: NDArray[np.float64],
                queries: NDArray[np.float64],
                matched_indices: NDArray[np.int64],
                normalized_mismatch: NDArray[np.float64],
                axis_spacing: float,
                max_examples: int = 5,
            ) -> None:
                diffs = np.diff(axis_values)
                is_ascending = np.all(diffs >= 0)
                is_descending = np.all(diffs <= 0)
                is_monotonic = is_ascending or is_descending

                n_low = np.sum(queries < np.min(axis_values))
                n_high = np.sum(queries > np.max(axis_values))

                logger.info(
                    f"{axis_name} axis diagnostics: monotonic={is_monotonic}, ascending={is_ascending}, "
                    f"axis_range=[{np.min(axis_values)}, {np.max(axis_values)}], "
                    f"query_range=[{np.min(queries)}, {np.max(queries)}], "
                    f"out_of_range={n_low + n_high} (low={n_low}, high={n_high}), "
                    f"typical_spacing={axis_spacing}."
                )

                q50, q90, q99 = np.quantile(normalized_mismatch, [0.5, 0.9, 0.99])
                logger.info(
                    f"{axis_name} spacing-normalized mismatch quantiles: q50={q50}, q90={q90}, q99={q99}, "
                    f"max={np.max(normalized_mismatch)}."
                )

                worst_local = np.argsort(normalized_mismatch)[-max_examples:][::-1]
                for rank, i in enumerate(worst_local, start=1):
                    matched_value = axis_values[matched_indices[i]]
                    abs_error = np.abs(queries[i] - matched_value)
                    norm_error = normalized_mismatch[i]
                    logger.info(
                        f"{axis_name} worst#{rank}: query={queries[i]}, matched={matched_value}, "
                        f"idx={matched_indices[i]}, "
                        f"abs_err={abs_error}, norm_err={norm_error}."
                    )

            logger.warning(
                f"{np.sum(mismatch_x > normalized_error_tol)} of {len(idx_x)} x-coordinates do not match within "
                "tolerance."
            )
            logger.warning(
                f"{np.sum(mismatch_y > normalized_error_tol)} of {len(idx_y)} y-coordinates do not match within "
                "tolerance."
            )
            logger.warning(
                f"Maximum spacing-normalized mismatch: {np.max(mismatch_x)} in x and {np.max(mismatch_y)} in y."
            )
            print_axis_diagnostics("x", x_axis, xs, idx_x, mismatch_x, x_spacing)

        surf = np.asarray(nc["surface"][:])
        result = surf[idx_y, idx_x]

    return result


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
    with open(dem_file) as f:
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
    parser.add_argument(
        "--elmer-mesh-crs-epsg",
        type=int,
        required=True,
        choices={3413, 3013},
        help="EPSG code of the input Elmer mesh coordinate reference system."
        " Used to convert mesh x/y coordinates to lon/lat.",
    )
    parser.add_argument("-o", "--outpath", type=Path, help="Output path to the new mesh with DEM.", default=None)
    parser.add_argument(
        "-i", "--in-place", help="Make changes to mesh in place (will overwrite existing mesh!)", action="store_true"
    )
    args = parser.parse_args()

    outpath: Path
    if args.in_place:
        assert args.outpath is None, "You cannot specify --outpath when using --in-place."
        outpath = args.elmer_mesh
    else:
        assert args.outpath is not None, "You must specify --outpath when not using --in-place."
        assert not args.outpath.exists(), (
            f"Output path {args.outpath} already exists. Please pick a different folder name or use the --in-place "
            f"option to overwrite the existing mesh at {args.elmer_mesh}."
        )
        outpath = args.outpath

    print("I'm running as main...")
    print(f"Reading the following files: {args.elmer_mesh} and {args.dem}")

    mesh = read_elmer_mesh(args.elmer_mesh, source_crs_epsg=args.elmer_mesh_crs_epsg)
    x = mesh.x_vertices
    y = mesh.y_vertices
    h = read_dem(args.dem, x, y)

    # Only copy when not operating in-place; skip nodes so we can write a fresh file
    if not args.in_place:
        assert args.elmer_mesh.is_dir(), f"{args.elmer_mesh} is no directory or does not exist."
        shutil.copytree(args.elmer_mesh, outpath, ignore=shutil.ignore_patterns("mesh.nodes"))

    write_dem_as_elmer(mesh, h, outpath / "mesh.nodes", allow_overwrite=args.in_place)
