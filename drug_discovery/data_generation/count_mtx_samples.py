#!/usr/bin/env python3

"""Count samples in a directory tree containing sparse `.mtx` files.

The project stores each split as a directory with paired matrices such as
`X_data.mtx` and `Y_data.mtx`. This script walks the tree recursively, counts
each split directory once, and sums the number of samples across all of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io


def count_samples_in_directory(root: Path) -> tuple[int, list[tuple[Path, int]]]:
    """Return the total number of samples and the per-directory counts."""

    split_dirs = sorted({path.parent for path in root.rglob("*.mtx")})
    per_dir_counts: list[tuple[Path, int]] = []
    total_samples = 0

    for split_dir in split_dirs:
        mtx_files = sorted(split_dir.glob("*.mtx"))
        if not mtx_files:
            continue

        row_counts = []
        for mtx_file in mtx_files:
            matrix = scipy.io.mmread(mtx_file)
            row_counts.append(matrix.shape[0])

        if len(set(row_counts)) != 1:
            raise ValueError(
                f"Inconsistent row counts in {split_dir}: {row_counts}"
            )

        sample_count = row_counts[0]
        per_dir_counts.append((split_dir, sample_count))
        total_samples += sample_count

    return total_samples, per_dir_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count samples across recursive .mtx split directories."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/student/Matan/Federated_Learning/drug_discovery/new/src/Datasets/data_40/full",
        help="Root directory containing recursive .mtx files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.path).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    total_samples, per_dir_counts = count_samples_in_directory(root)

    print(f"Root: {root}")
    print(f"Split directories found: {len(per_dir_counts)}")
    for split_dir, sample_count in per_dir_counts:
        print(f"{split_dir}: {sample_count} samples")
    print(f"Total samples: {total_samples}")


if __name__ == "__main__":
    main()