"""CLI entry for leakage-safe feature engineering of split standard spectral packages."""

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
        candidates.append(parent / "spectral-core")
    for candidate in candidates:
        if (candidate / "spectral_core" / "__init__.py").is_file():
            return candidate
    return here.parents[3]


ROOT = _find_runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.feature.workflow import feature_spectral_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature-engineer a standard spectral package using train-only fitted rules.")
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--preprocess-contract", help="Fold/repeat-wise preprocess_contract.json to consume directly.")
    parser.add_argument("--split-contract")
    parser.add_argument("--output-dir")
    parser.add_argument("--method")
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--explained-variance", type=float)
    parser.add_argument("--variance-threshold", type=float)
    parser.add_argument("--band-min", type=float)
    parser.add_argument("--band-max", type=float)
    parser.add_argument("--band-indices")
    parser.add_argument("--feature-names")
    parser.add_argument("--index-base", type=int, choices=[0, 1], default=0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--n-intervals", type=int)
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--cv", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--task-type", choices=["classification", "regression"])
    parser.add_argument("--correlation-method", choices=["pearson", "spearman"])
    parser.add_argument("--interval-mode")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--device", help="cpu, auto, cuda, or cuda:<index>.")
    parser.add_argument("--feature-config")
    parser.add_argument("--auto-confirm-feature-defaults", action="store_true")
    parser.add_argument("--confirm-unsplit-fit", action="store_true")
    parser.add_argument("--confirm-deep-embedding-training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = feature_spectral_package(
        package_dir=args.package_dir,
        preprocess_contract=args.preprocess_contract,
        split_contract=args.split_contract,
        output_dir=args.output_dir,
        method=args.method,
        n_components=args.n_components,
        explained_variance=args.explained_variance,
        variance_threshold=args.variance_threshold,
        band_min=args.band_min,
        band_max=args.band_max,
        band_indices=args.band_indices,
        feature_names=args.feature_names,
        index_base=args.index_base,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        n_intervals=args.n_intervals,
        n_runs=args.n_runs,
        sample_ratio=args.sample_ratio,
        cv=args.cv,
        random_state=args.random_state,
        task_type=args.task_type,
        correlation_method=args.correlation_method,
        interval_mode=args.interval_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        noise_std=args.noise_std,
        mask_ratio=args.mask_ratio,
        temperature=args.temperature,
        patch_size=args.patch_size,
        device=args.device,
        feature_config=args.feature_config,
        auto_confirm_feature_defaults=args.auto_confirm_feature_defaults,
        confirm_unsplit_fit=args.confirm_unsplit_fit,
        confirm_deep_embedding_training=args.confirm_deep_embedding_training,
        overwrite=args.overwrite,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
