#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Compare two EBFM reference snapshots produced by --dump-reference.

Usage:
    python tools/compare_snapshots.py baseline.npz candidate.npz
    python tools/compare_snapshots.py baseline.npz candidate.npz --atol 1e-10 --rtol 1e-6
    python tools/compare_snapshots.py baseline.npz candidate.npz --no-fail

Exit codes:
    0  all arrays are within tolerance (or --no-fail is set)
    1  one or more arrays exceed tolerance
"""

import argparse
import sys

import numpy as np


def compare(baseline_path: str, candidate_path: str, atol: float, rtol: float, no_fail: bool) -> bool:
    baseline = np.load(baseline_path)
    candidate = np.load(candidate_path)

    all_keys = sorted(set(baseline.files) | set(candidate.files))
    only_in_baseline = sorted(set(baseline.files) - set(candidate.files))
    only_in_candidate = sorted(set(candidate.files) - set(baseline.files))
    common_keys = sorted(set(baseline.files) & set(candidate.files))

    print(f"Baseline : {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Tolerance: atol={atol:.2e}, rtol={rtol:.2e}")
    print()

    if only_in_baseline:
        print(f"  [WARN] Keys only in baseline (missing from candidate): {only_in_baseline}")
    if only_in_candidate:
        print(f"  [WARN] Keys only in candidate (not in baseline):        {only_in_candidate}")
    if only_in_baseline or only_in_candidate:
        print()

    any_failed = False

    col_w = max(len(k) for k in all_keys) + 2

    header = (
        f"  {'Array':<{col_w}}  {'Shape':<20}  {'Max |diff|':>14}  "
        f"{'Max rel diff':>14}  {'Mean |diff|':>14}  {'Status':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key in common_keys:
        b = baseline[key]
        c = candidate[key]

        if b.shape != c.shape:
            print(f"  {key:<{col_w}}  SHAPE MISMATCH: baseline={b.shape} candidate={c.shape}")
            any_failed = True
            continue

        # General case: exact integer comparison for all integer-typed arrays
        if np.issubdtype(b.dtype, np.integer) and np.issubdtype(c.dtype, np.integer):
            exact_equal = np.array_equal(b, c)
            abs_diff = np.abs(c - b)
            max_abs = abs_diff.max() if b.size > 0 else 0
            mean_abs = int(abs_diff.mean()) if b.size > 0 else 0
            # For relative difference, cast to float for division, but show as float
            denom = np.where(np.abs(b) > 0, np.abs(b), 1)
            max_rel = float((abs_diff / denom).max()) if b.size > 0 else 0.0
            passed = exact_equal
            status = "OK" if passed else "FAIL"
            if not passed:
                any_failed = True
            print(
                f"  {key:<{col_w}}  {str(b.shape):<20}  {max_abs:>14}  " f"{max_rel:>14}  {mean_abs:>14}  {status:>8}"
            )
            continue

        # Default: float/tolerance-based comparison
        b = b.astype(float)
        c = c.astype(float)
        abs_diff = np.abs(c - b)
        max_abs = abs_diff.max()
        mean_abs = abs_diff.mean()
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        max_rel = (abs_diff / denom).max()
        passed = bool(max_abs <= atol + rtol * np.abs(b).max())
        status = "OK" if passed else "FAIL"
        if not passed:
            any_failed = True
        print(
            f"  {key:<{col_w}}  {str(b.shape):<20}  {max_abs:>14.4e}  "
            f"{max_rel:>14.4e}  {mean_abs:>14.4e}  {status:>8}"
        )

    print()
    if any_failed:
        print("Result: DIFFERENCES EXCEED TOLERANCE")
    else:
        print("Result: ALL ARRAYS WITHIN TOLERANCE")

    return not any_failed


def main():
    parser = argparse.ArgumentParser(
        description="Compare two EBFM --dump-reference snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline", help="Path to the baseline .npz snapshot.")
    parser.add_argument("candidate", help="Path to the candidate .npz snapshot to compare against baseline.")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for pass/fail.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for pass/fail.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        default=False,
        help="Always exit 0 even if differences exceed tolerance (print-only mode).",
    )

    args = parser.parse_args()
    passed = compare(args.baseline, args.candidate, args.atol, args.rtol, args.no_fail)

    if args.no_fail:
        sys.exit(0)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
