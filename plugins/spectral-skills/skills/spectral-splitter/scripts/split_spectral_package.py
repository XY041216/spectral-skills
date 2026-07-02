"""CLI entry for compact train/validation/test splitting of standard spectral packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.splitter.workflow import split_spectral_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a standard spectral package without rewriting spectral data.")
    parser.add_argument("--package-dir", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--method",
        choices=[
            "auto",
            "random",
            "stratified",
            "predefined_split",
            "kfold",
            "stratified_kfold",
            "leave_one_out",
            "monte_carlo_cv",
            "repeated_random_split",
            "stratified_monte_carlo_cv",
            "kennard_stone",
            "spxy",
            "duplex",
            "regression_stratified",
            "y_binned_stratified",
            "group",
            "group_aware",
            "stratified_group",
        ],
        default="auto",
    )
    parser.add_argument("--ratio", help="Ratio string such as 8:2, 7:3, or 6:2:2.")
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--split-column")
    parser.add_argument("--split-indices-file")
    parser.add_argument("--group-column")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--scale", default="standardize", choices=["standardize", "none"])
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--confirm-stratified", action="store_true")
    parser.add_argument("--confirm-incomplete-ratio", action="store_true", help="User confirmed how to complete an incomplete ratio such as 6:2:.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = split_spectral_package(
        package_dir=args.package_dir,
        output_dir=args.output_dir,
        method=args.method,
        ratio=args.ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_column=args.split_column,
        split_indices_file=args.split_indices_file,
        group_column=args.group_column,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        shuffle=not args.no_shuffle,
        n_bins=args.n_bins,
        scale=args.scale,
        random_seed=args.random_seed,
        confirm_stratified=args.confirm_stratified,
        confirm_incomplete_ratio=args.confirm_incomplete_ratio,
        overwrite=args.overwrite,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
