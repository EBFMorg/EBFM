# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""What a restart file contains, and the checks it has to satisfy.

Single source of truth for the state carried across a restart, shared by the writer
(`FINAL_create_restart_file`) and the loader (`INIT.init_initial_conditions`) so the two cannot
drift apart.
"""

from pathlib import Path

# One value per snow/firn column: shape `(gpsum,)`.
PER_COLUMN_VARIABLES = (
    "snowmass",
    "Tsurf",
    "ys",
    "timelastsnow_netCDF",
    "alb_snow",
    "surface_elevation",
    "x",
    "y",
)

# One value per layer of every column: shape `(gpsum, nl)`.
PER_LAYER_VARIABLES = ("subZ", "subW", "subD", "subS", "subT", "subTmean")

# Every variable a restart file carries. The order is the order they are written in.
RESTART_VARIABLES = PER_LAYER_VARIABLES + PER_COLUMN_VARIABLES


def expected_shape(var_name: str, gpsum: int, nl: int) -> tuple[int, ...] | None:
    """
    The shape a named restart variable must have.

    Parameters:
        var_name (str): Name of the variable as stored in the restart file
        gpsum (int): Number of columns
        nl (int): Number of layers per column

    Returns:
        tuple | None: the required shape, or None for a variable outside the manifest
    """
    if var_name in PER_LAYER_VARIABLES:
        return (gpsum, nl)
    if var_name in PER_COLUMN_VARIABLES:
        return (gpsum,)
    return None


def validate_variable_shape(var_name: str, shape: tuple[int, ...], gpsum: int, nl: int, source: Path) -> None:
    """
    Check one restart variable against the configured column discretization.

    Variables in the manifest must have exactly the shape their group requires, which is what rules
    out a per-layer array stored per-column or vice versa. Anything else only has to carry a
    recognisable column axis, so a restart file may gain auxiliary variables without changes here.

    Parameters:
        var_name (str): Name of the variable as stored in the restart file
        shape (tuple): Shape the variable actually has in the file
        gpsum (int): Number of columns
        nl (int): Number of layers per column
        source (Path): Restart file the variable came from, for the error message

    Raises:
        ValueError: if the shape does not match
    """
    required = expected_shape(var_name, gpsum, nl)
    if required is not None:
        if shape != required:
            raise ValueError(
                f"Restart variable '{var_name}' in {source} has shape {shape}, " f"but must have shape {required}."
            )
        return

    if len(shape) > 2:
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has {len(shape)} dimensions; "
            "restart files hold per-column and per-layer variables only."
        )
    if len(shape) == 1 and shape != (gpsum,):
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has shape {shape}, "
            f"but every per-column variable must have shape {(gpsum,)}."
        )
    if len(shape) == 2 and shape != (gpsum, nl):
        raise ValueError(
            f"Restart variable '{var_name}' in {source} has shape {shape}, "
            f"but every per-layer variable must have shape {(gpsum, nl)}."
        )


def validate_all_variables_present(var_names, source: Path) -> None:
    """
    Check that a restart file carries every variable the model needs to resume.

    A missing variable is otherwise only noticed when the time loop indexes it, far from the cause.

    Parameters:
        var_names (Iterable[str]): Names present in the restart file
        source (Path): Restart file being loaded, for the error message

    Raises:
        ValueError: if any manifest variable is absent
    """
    missing = [name for name in RESTART_VARIABLES if name not in set(var_names)]
    if missing:
        raise ValueError(f"Restart file {source} is missing the required variable(s): {', '.join(missing)}.")
