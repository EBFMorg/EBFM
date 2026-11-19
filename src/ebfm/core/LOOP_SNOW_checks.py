# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input checks of LOOP_SNOW.

A corrupted column stops the run with a message naming the column, its location and the offending layers, instead of
spreading silently.
"""

import numpy as np
from numpy.typing import NDArray

from ebfm.core.constants import MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE

_MAX_REPORTED_COLUMNS = 5


def _column_label(grid, i: int) -> str:
    if "lat" in grid and "lon" in grid:
        return f"column {i} (lat={grid['lat'][i]:.4f}, lon={grid['lon'][i]:.4f})"
    return f"column {i}"


def _heat_conduction_nonfinite(OUT) -> dict[str, NDArray[np.bool_]]:
    """Per field, a (gpsum, nl) mask of the non-finite values heat conduction reads.

    subT of layer 0 is not read: the surface ghost layer takes Tsurf instead.
    """
    subT_nonfinite = ~np.isfinite(OUT["subT"])
    subT_nonfinite[:, 0] = False
    return {
        "Tsurf": ~np.isfinite(OUT["Tsurf"])[:, np.newaxis],
        "subT": subT_nonfinite,
        "subD": ~np.isfinite(OUT["subD"]),
        "subZ": ~np.isfinite(OUT["subZ"]),
    }


def heat_conduction_inputs_finite(OUT) -> bool:
    return (
        np.isfinite(OUT["Tsurf"]).all()
        and np.isfinite(OUT["subT"][:, 1:]).all()
        and np.isfinite(OUT["subD"]).all()
        and np.isfinite(OUT["subZ"]).all()
    )


def _describe_nonfinite_heat_conduction_inputs(OUT, grid) -> str:
    nonfinite = _heat_conduction_nonfinite(OUT)
    bad = np.flatnonzero(np.any([mask.any(axis=1) for mask in nonfinite.values()], axis=0))
    lines = [
        f"Non-finite heat conduction input in {bad.size} of {OUT['Tsurf'].size} columns. "
        "Check the forcing, the coupling fields and the restart file."
    ]
    for i in bad[:_MAX_REPORTED_COLUMNS]:
        causes = []
        if nonfinite["Tsurf"][i, 0]:
            causes.append(f"Tsurf={OUT['Tsurf'][i]}")
        for name in ("subT", "subD", "subZ"):
            layers = np.flatnonzero(nonfinite[name][i])
            if layers.size:
                causes.append(f"{name} non-finite in layers {layers.tolist()}")
        lines.append(f"  {_column_label(grid, i)}: " + "; ".join(causes))
    if bad.size > _MAX_REPORTED_COLUMNS:
        lines.append(f"  ... and {bad.size - _MAX_REPORTED_COLUMNS} more columns")
    return "\n".join(lines)


def _describe_implausible_subT(OUT, grid) -> str:
    """Explain the columns with a layer 1..nl-1 colder than MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE."""
    subT = OUT["subT"][:, 1:]
    bad = np.flatnonzero((subT < MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE).any(axis=1))
    lines = [
        f"subT < {MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE:g} K in {bad.size} of {subT.shape[0]} columns. "
        "Temperatures in deg C or fill values in the forcing, the coupling fields or the restart file?"
    ]
    for i in bad[:_MAX_REPORTED_COLUMNS]:
        layers = np.flatnonzero(subT[i] < MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE) + 1
        lines.append(
            f"  {_column_label(grid, i)}: Tsurf={OUT['Tsurf'][i]:.6g}; "
            f"layers {layers.tolist()}, min {subT[i].min():.6g} K"
        )
    if bad.size > _MAX_REPORTED_COLUMNS:
        lines.append(f"  ... and {bad.size - _MAX_REPORTED_COLUMNS} more columns")
    return "\n".join(lines)


def assert_heat_conduction_inputs(OUT, grid) -> None:
    assert heat_conduction_inputs_finite(OUT), _describe_nonfinite_heat_conduction_inputs(OUT, grid)
    assert (OUT["subT"][:, 1:] >= MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE).all(), _describe_implausible_subT(OUT, grid)


def describe_dt_stab_failure(dt_stab: NDArray[np.float64], OUT, grid) -> str:
    """Explain the columns where dt_stab is not > 0, using the layers 1..nl-1 that enter it."""
    bad = np.flatnonzero(~(dt_stab > 0))
    lines = [f"dt_stab <= 0 or NaN in {bad.size} of {dt_stab.size} columns."]
    for i in bad[:_MAX_REPORTED_COLUMNS]:
        subT = OUT["subT"][i, 1:]
        subD = OUT["subD"][i, 1:]
        subZ = OUT["subZ"][i, 1:]
        causes = []
        no_density = np.flatnonzero(subD <= 0)
        if no_density.size:
            causes.append(f"subD <= 0 in layers {(no_density + 1).tolist()}, min {subD.min():.6g}")
        no_thickness = np.flatnonzero(subZ <= 0)
        if no_thickness.size:
            causes.append(f"subZ <= 0 in layers {(no_thickness + 1).tolist()}, min {subZ.min():.6g}")
        if not causes:
            causes.append(
                f"no layer out of range: subT in [{subT.min():.6g}, {subT.max():.6g}], "
                f"subD in [{subD.min():.6g}, {subD.max():.6g}], subZ in [{subZ.min():.6g}, {subZ.max():.6g}]"
            )
        lines.append(
            f"  {_column_label(grid, i)}: dt_stab={dt_stab[i]:.6g}, Tsurf={OUT['Tsurf'][i]:.6g}; " + "; ".join(causes)
        )
    if bad.size > _MAX_REPORTED_COLUMNS:
        lines.append(f"  ... and {bad.size - _MAX_REPORTED_COLUMNS} more columns")
    return "\n".join(lines)
