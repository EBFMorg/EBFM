# SPDX-FileCopyrightText: 2026 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input checks and iteration limits of LOOP_SNOW.

A corrupted column stops the run with a message naming the column, its location and the offending layers, instead of
spreading silently or hanging a loop of the snow and firn model.
"""

import numpy as np
from numpy.typing import NDArray

from ebfm.core import logging
from ebfm.core.constants import MIN_PLAUSIBLE_FRESH_SNOW_DENSITY, MIN_PLAUSIBLE_SUBSURFACE_TEMPERATURE

logger = logging.getLogger(__name__)

# Iteration limits far above any physical column. A corrupted column stops the run instead of hanging it.
MAX_HEAT_SUBSTEPS = 1_000_000  # Heat-conduction sub-steps per column and time step, usually about 10
MAX_SHIFT_PASSES = 1000  # Snowfall or melt grid-shift passes per time step, usually a few

# Heat-conduction stability time steps below this get a warning; usually they are a minute or more.
WARN_DT_STAB_SECONDS = 1.0

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


def _report_columns(header: str, bad: NDArray[np.intp], grid, describe_column) -> str:
    """header, then one line per bad column (at most _MAX_REPORTED_COLUMNS) from describe_column(i)."""
    lines = [header]
    for i in bad[:_MAX_REPORTED_COLUMNS]:
        lines.append(f"  {_column_label(grid, i)}: {describe_column(i)}")
    if bad.size > _MAX_REPORTED_COLUMNS:
        lines.append(f"  ... and {bad.size - _MAX_REPORTED_COLUMNS} more columns")
    return "\n".join(lines)


def _too_many_heat_substeps(dt_stab: NDArray[np.float64], dt: float) -> NDArray[np.bool_]:
    """Columns whose dt / dt_stab sub-steps exceed MAX_HEAT_SUBSTEPS. Takes dt_stab > 0."""
    return dt_stab < dt / MAX_HEAT_SUBSTEPS


def _describe_too_many_heat_substeps(dt_stab: NDArray[np.float64], dt: float, OUT, grid) -> str:
    bad = np.flatnonzero(_too_many_heat_substeps(dt_stab, dt))

    def describe(i):
        subZ = OUT["subZ"][i, 1:]
        subD = OUT["subD"][i, 1:]
        thinnest = int(np.argmin(subZ))
        lightest = int(np.argmin(subD))
        return (
            f"dt_stab={dt_stab[i]:.6g} d ({dt / dt_stab[i]:.3g} sub-steps); "
            f"thinnest layer {thinnest + 1}: subZ={subZ[thinnest]:.6g} m; "
            f"lightest layer {lightest + 1}: subD={subD[lightest]:.6g} kg m-3"
        )

    return _report_columns(
        f"Heat conduction needs more than {MAX_HEAT_SUBSTEPS} sub-steps in {bad.size} of {dt_stab.size} columns. "
        "A layer of near-zero thickness or density?",
        bad,
        grid,
        describe,
    )


def assert_dt_stab(dt_stab: NDArray[np.float64], dt: float, OUT, grid) -> None:
    assert (dt_stab > 0).all(), describe_dt_stab_failure(dt_stab, OUT, grid)
    assert not _too_many_heat_substeps(dt_stab, dt).any(), _describe_too_many_heat_substeps(dt_stab, dt, OUT, grid)


def warn_small_dt_stab(dt_stab: NDArray[np.float64], dt: float, dayseconds: float, OUT, grid) -> None:
    """Warn if dt_stab (d) of a column is below WARN_DT_STAB_SECONDS. Takes dt_stab > 0."""
    small = dt_stab < WARN_DT_STAB_SECONDS / dayseconds
    if not small.any():
        return
    i = int(np.argmin(dt_stab))
    subZ = OUT["subZ"][i, 1:]
    subD = OUT["subD"][i, 1:]
    thinnest = int(np.argmin(subZ))
    lightest = int(np.argmin(subD))
    logger.warning(
        f"Heat conduction stability time step below {WARN_DT_STAB_SECONDS:g} s in {int(small.sum())} of "
        f"{dt_stab.size} columns, so the solve needs many sub-steps and runs slowly. Smallest in "
        f"{_column_label(grid, i)}: dt_stab={dt_stab[i] * dayseconds:.3g} s ({dt / dt_stab[i]:.3g} sub-steps); "
        f"thinnest layer {thinnest + 1}: subZ={subZ[thinnest]:.6g} m; "
        f"lightest layer {lightest + 1}: subD={subD[lightest]:.6g} kg m-3. "
        f"More than {MAX_HEAT_SUBSTEPS} sub-steps stop the run."
    )


def _describe_implausible_fresh_snow_density(OUT, IN, grid) -> str:
    bad = np.flatnonzero(~(OUT["Dfreshsnow"] >= MIN_PLAUSIBLE_FRESH_SNOW_DENSITY))
    return _report_columns(
        f"Fresh snow density < {MIN_PLAUSIBLE_FRESH_SNOW_DENSITY:g} kg m-3 or NaN in {bad.size} of "
        f"{OUT['Dfreshsnow'].size} columns. Implausible air temperature or wind speed in the forcing or the coupling "
        "fields?",
        bad,
        grid,
        lambda i: f"Dfreshsnow={OUT['Dfreshsnow'][i]:.6g} kg m-3, T={IN['T'][i]:.6g} K, WS={IN['WS'][i]:.6g} m s-1",
    )


def assert_fresh_snow_density(OUT, IN, grid) -> None:
    assert (OUT["Dfreshsnow"] >= MIN_PLAUSIBLE_FRESH_SNOW_DENSITY).all(), _describe_implausible_fresh_snow_density(
        OUT, IN, grid
    )


def _too_many_snowfall_passes(shift_tot: NDArray[np.float64], max_subZ: float) -> NDArray[np.bool_]:
    """Columns whose fresh snow (m) needs more than MAX_SHIFT_PASSES passes of max_subZ, or is NaN."""
    return ~(shift_tot <= MAX_SHIFT_PASSES * max_subZ)


def _describe_too_many_snowfall_passes(shift_tot: NDArray[np.float64], max_subZ: float, OUT, IN, grid) -> str:
    bad = np.flatnonzero(_too_many_snowfall_passes(shift_tot, max_subZ))
    return _report_columns(
        f"Fresh snow is thicker than {MAX_SHIFT_PASSES} top layers of {max_subZ:g} m, or NaN, in {bad.size} of "
        f"{shift_tot.size} columns. Implausible snowfall, deposition or fresh snow density?",
        bad,
        grid,
        lambda i: (
            f"fresh snow {shift_tot[i]:.6g} m from snow={IN['snow'][i]:.6g} and "
            f"moist_deposition={OUT['moist_deposition'][i]:.6g} m w.e., Dfreshsnow={OUT['Dfreshsnow'][i]:.6g} kg m-3"
        ),
    )


def assert_snowfall_passes(shift_tot: NDArray[np.float64], max_subZ: float, OUT, IN, grid) -> None:
    assert not _too_many_snowfall_passes(shift_tot, max_subZ).any(), _describe_too_many_snowfall_passes(
        shift_tot, max_subZ, OUT, IN, grid
    )


def describe_column_melted_away(mass_removed: NDArray[np.float64], mass_layer: NDArray[np.float64], OUT, grid) -> str:
    bad = np.flatnonzero(mass_removed > 0)
    return _report_columns(
        f"Melt and sublimation remove more mass than the whole column holds in {bad.size} of {mass_removed.size} "
        "columns. Implausible melt or sublimation?",
        bad,
        grid,
        lambda i: (
            f"melt={OUT['melt'][i]:.6g}, moist_sublimation={OUT['moist_sublimation'][i]:.6g} m w.e., "
            f"column mass {mass_layer[i].sum():.6g} kg m-2"
        ),
    )


def describe_too_many_melt_passes(shift_tot: NDArray[np.float64], OUT, grid) -> str:
    bad = np.flatnonzero(shift_tot < 0)
    return _report_columns(
        f"Melt removal did not finish within {MAX_SHIFT_PASSES} grid-shift passes in {bad.size} of {shift_tot.size} "
        "columns. A layer 1 of near-zero thickness?",
        bad,
        grid,
        lambda i: f"remaining shift {shift_tot[i]:.6g} m, subZ[0]={OUT['subZ'][i, 0]:.6g} m, "
        f"subZ[1]={OUT['subZ'][i, 1]:.6g} m",
    )
