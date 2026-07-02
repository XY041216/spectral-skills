"""CLI entry for compact spectral QC on reader standard packages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_runtime_root() -> Path:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent)
    for candidate in candidates:
        if (candidate / "spectral_core" / "__init__.py").is_file():
            return candidate
    return here.parents[3]


ROOT = _find_runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.qc.workflow import qc_spectral_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compact QC checks or confirmed actions on a standard spectral package.")
    parser.add_argument("--package-dir", default=".")
    parser.add_argument("--mode", choices=["methods", "check", "mark", "clean", "outliers", "apply"], default="check")
    parser.add_argument("--methods", help="Comma-separated method IDs such as MD,PCA_DISTANCE,IQR.")
    parser.add_argument("--output-dir")
    parser.add_argument("--confirm-action", action="store_true")
    parser.add_argument("--remove-sample-ids")
    parser.add_argument("--remove-sample-indices")
    parser.add_argument("--remove-band-indices")
    parser.add_argument("--impute-missing", choices=["none", "mean", "median", "zero", "linear", "nearest"])
    parser.add_argument("--cleaning-action")
    parser.add_argument("--cleaning-method")
    parser.add_argument("--cleaning-strategy")
    parser.add_argument("--threshold")
    parser.add_argument("--n-resamples", type=int)
    parser.add_argument("--sample-fraction", type=float)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--base-model")
    parser.add_argument("--outlier-metric")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--detail-level", choices=["summary", "full"], default="summary")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export-details", action="store_true")
    args = parser.parse_args(argv)
    response = qc_spectral_package(
        package_dir=args.package_dir,
        mode=args.mode,
        methods=_split_csv(args.methods),
        output_dir=args.output_dir,
        confirm_action=args.confirm_action,
        remove_sample_ids=_split_csv(args.remove_sample_ids),
        remove_sample_indices=_split_ints(args.remove_sample_indices),
        remove_band_indices=_split_ints(args.remove_band_indices),
        impute_missing=args.impute_missing,
        cleaning_action=args.cleaning_action,
        cleaning_method=args.cleaning_method,
        cleaning_strategy=args.cleaning_strategy,
        threshold=_parse_threshold(args.threshold),
        n_resamples=args.n_resamples,
        sample_fraction=args.sample_fraction,
        train_ratio=args.train_ratio,
        base_model=args.base_model,
        outlier_metric=args.outlier_metric,
        detail_level="full" if args.verbose else args.detail_level,
        export_details=args.export_details,
        overwrite=args.overwrite,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_threshold(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
