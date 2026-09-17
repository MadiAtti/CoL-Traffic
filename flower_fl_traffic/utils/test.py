"""
compare_parquet.py
------------------
Utilities to compare two Parquet files.

Quick-check phase  : schema labels + row/column counts (cheap, O(1) metadata)
Deep-check phase   : full row-by-row diff  (only runs if quick check passes)

Usage
-----
    result = compare_parquet("file_a.parquet", "file_b.parquet")
    if result["equal"]:
        print("Files are identical.")
    else:
        print(result["summary"])
        print(result["details"])   # DataFrame of differing rows (deep check only)
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CompareResult:
    equal: bool
    phase: str                              # "quick" or "deep"
    summary: str                            # human-readable verdict
    schema_a: Optional[object] = None
    schema_b: Optional[object] = None
    details: Optional[pd.DataFrame] = None  # differing rows (deep check only)
    diff_count: int = 0
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – quick check (labels + size)
# ──────────────────────────────────────────────────────────────────────────────

def quick_check(path_a: str, path_b: str) -> CompareResult:
    """
    Compare schema (column names + dtypes) and row/column counts.
    Fast: reads only Parquet metadata, not the actual data.

    Returns
    -------
    CompareResult  with phase="quick"
    """
    meta_a = pq.read_metadata(path_a)
    meta_b = pq.read_metadata(path_b)

    schema_a = pq.read_schema(path_a)
    schema_b = pq.read_schema(path_b)

    errors: list[str] = []

    # ── row count ──────────────────────────────────────────────────────────────
    rows_a = meta_a.num_rows
    rows_b = meta_b.num_rows
    if rows_a != rows_b:
        errors.append(f"Row count differs: {rows_a} vs {rows_b}")

    # ── column count ───────────────────────────────────────────────────────────
    cols_a = schema_a.names
    cols_b = schema_b.names
    if len(cols_a) != len(cols_b):
        errors.append(
            f"Column count differs: {len(cols_a)} vs {len(cols_b)}"
        )

    # ── column labels ─────────────────────────────────────────────────────────
    only_in_a = set(cols_a) - set(cols_b)
    only_in_b = set(cols_b) - set(cols_a)
    if only_in_a:
        errors.append(f"Columns only in A: {sorted(only_in_a)}")
    if only_in_b:
        errors.append(f"Columns only in B: {sorted(only_in_b)}")

    # ── column order ───────────────────────────────────────────────────────────
    if cols_a != cols_b and not (only_in_a or only_in_b):
        errors.append(f"Column order differs:\n  A: {cols_a}\n  B: {cols_b}")

    # ── data types ────────────────────────────────────────────────────────────
    common_cols = [c for c in cols_a if c in set(cols_b)]
    for col in common_cols:
        t_a = schema_a.field(col).type
        t_b = schema_b.field(col).type
        if t_a != t_b:
            errors.append(f"dtype mismatch on '{col}': {t_a} vs {t_b}")

    equal = len(errors) == 0
    summary = (
        "Quick check PASSED – schema and size match."
        if equal
        else "Quick check FAILED:\n  " + "\n  ".join(errors)
    )

    return CompareResult(
        equal=equal,
        phase="quick",
        summary=summary,
        schema_a=schema_a,
        schema_b=schema_b,
        errors=errors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 – deep check (row by row)
# ──────────────────────────────────────────────────────────────────────────────

def deep_check(
    path_a: str,
    path_b: str,
    check_order: bool = True,
    sort_by: Optional[list[str]] = None,
    float_tol: float = 1e-9,
) -> CompareResult:
    """
    Load both files and compare every cell.

    Parameters
    ----------
    path_a, path_b : str
        Paths to the Parquet files.
    check_order : bool
        If True (default) rows must appear in the same order.
        If False, both DataFrames are sorted before comparison.
    sort_by : list[str] | None
        Column(s) to sort on when check_order=False.
        If None and check_order=False, sorts on all columns.
    float_tol : float
        Absolute tolerance for floating-point comparisons (via pandas.testing).

    Returns
    -------
    CompareResult  with phase="deep"
    """
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    errors: list[str] = []

    # ── align columns ─────────────────────────────────────────────────────────
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    df_a = df_a[common_cols].reset_index(drop=True)
    df_b = df_b[common_cols].reset_index(drop=True)

    # ── optional sort ─────────────────────────────────────────────────────────
    if not check_order:
        sort_keys = sort_by or common_cols
        sort_keys = [k for k in sort_keys if k in common_cols]
        df_a = df_a.sort_values(sort_keys).reset_index(drop=True)
        df_b = df_b.sort_values(sort_keys).reset_index(drop=True)

    # ── row count guard ───────────────────────────────────────────────────────
    if len(df_a) != len(df_b):
        errors.append(f"Row count differs: {len(df_a)} vs {len(df_b)}")
        return CompareResult(
            equal=False,
            phase="deep",
            summary="Deep check FAILED:\n  " + "\n  ".join(errors),
            errors=errors,
        )

    # ── cell-level diff ───────────────────────────────────────────────────────
    diff_mask = pd.DataFrame(False, index=df_a.index, columns=common_cols)

    for col in common_cols:
        col_a = df_a[col]
        col_b = df_b[col]

        # float columns: use tolerance
        if pd.api.types.is_float_dtype(col_a) or pd.api.types.is_float_dtype(col_b):
            diff_mask[col] = ~(
                (col_a - col_b).abs().le(float_tol)
                | (col_a.isna() & col_b.isna())
            )
        else:
            diff_mask[col] = col_a.ne(col_b) & ~(col_a.isna() & col_b.isna())

    rows_with_diffs = diff_mask.any(axis=1)
    diff_count = int(rows_with_diffs.sum())

    # ── build diff report ─────────────────────────────────────────────────────
    details: Optional[pd.DataFrame] = None
    if diff_count > 0:
        # Produce a side-by-side report for differing rows
        idx = df_a.index[rows_with_diffs]
        rows_a = df_a.loc[idx].add_suffix("_A")
        rows_b = df_b.loc[idx].add_suffix("_B")
        details = pd.concat([rows_a, rows_b], axis=1)
        # Interleave A/B columns for readability
        interleaved = []
        for col in common_cols:
            if diff_mask.loc[idx, col].any():
                interleaved += [f"{col}_A", f"{col}_B"]
        details = details[interleaved]
        details.index.name = "row_index"

        cols_with_diffs = [c for c in common_cols if diff_mask[c].any()]
        errors.append(
            f"{diff_count} row(s) differ across "
            f"{len(cols_with_diffs)} column(s): {cols_with_diffs}"
        )

    equal = diff_count == 0
    summary = (
        "Deep check PASSED – all rows and cells match."
        if equal
        else "Deep check FAILED:\n  " + "\n  ".join(errors)
    )

    return CompareResult(
        equal=equal,
        phase="deep",
        summary=summary,
        details=details,
        diff_count=diff_count,
        errors=errors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point: quick first, deep only if quick passes
# ──────────────────────────────────────────────────────────────────────────────

def compare_parquet(
    path_a: str,
    path_b: str,
    *,
    check_order: bool = True,
    sort_by: Optional[list[str]] = None,
    float_tol: float = 1e-9,
    verbose: bool = True,
) -> CompareResult:
    """
    Compare two Parquet files in two phases:

    1. **Quick check** – schema (column names + dtypes) and row/column counts.
       Reads only Parquet metadata, very fast even for huge files.

    2. **Deep check** – row-by-row, cell-by-cell comparison.
       Only runs when the quick check passes (same schema and size).

    Parameters
    ----------
    path_a, path_b : str
        Paths to the two Parquet files to compare.
    check_order : bool
        Whether row order matters (default True).
        Set to False to sort before comparing (requires sort_by or sorts all cols).
    sort_by : list[str] | None
        Columns to sort on when check_order=False.
    float_tol : float
        Absolute tolerance for float comparisons.
    verbose : bool
        Print a summary to stdout.

    Returns
    -------
    CompareResult
        .equal      – True only when both phases pass
        .phase      – which phase produced the final verdict
        .summary    – human-readable explanation
        .details    – DataFrame of differing rows (deep phase only, or None)
        .diff_count – number of differing rows (deep phase only)
        .errors     – list of individual error strings
    """
    # ── Phase 1 ───────────────────────────────────────────────────────────────
    qr = quick_check(path_a, path_b)
    if verbose:
        print(f"[Phase 1 – Quick] {qr.summary}")

    if not qr.equal:
        return qr  # no point loading data if schema/size differ

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    dr = deep_check(
        path_a, path_b,
        check_order=check_order,
        sort_by=sort_by,
        float_tol=float_tol,
    )
    if verbose:
        print(f"[Phase 2 – Deep ] {dr.summary}")
        if dr.details is not None:
            print(f"\nFirst 10 differing rows:\n{dr.details.head(10)}")

    return dr


# ──────────────────────────────────────────────────────────────────────────────
# Example / smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("=== Comparing identical files ===")
    r = compare_parquet("dataset/dataset.parquet", "dataset/other.parquet")
    print(f"Result: equal={r.equal}\n")

    dataset = pd.read_parquet("dataset/dataset.parquet")
    other = pd.read_parquet("dataset/other.parquet")
    print(dataset.shape)
    print(other.shape)
    print(dataset.head(5))
    print(other.head(5))