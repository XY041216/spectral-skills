"""CLI entry for compact spectral skill-chain orchestration."""

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

from spectral_core.workflow.workflow import run_spectral_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal spectral workflow across reader, splitter, preprocess, feature, and modeling.")
    parser.add_argument("--input")
    parser.add_argument("--package-dir")
    parser.add_argument("--data-contract")
    parser.add_argument("--split-contract")
    parser.add_argument("--output-dir", default=None, help="Explicit single run directory. Prefer --output-root for managed spectral_runs layout.")
    parser.add_argument("--output-root", help="Managed run root, for example data/spectral_runs. Workflow creates <output-root>/<dataset>/<run_id>.")
    parser.add_argument("--run-name", help="Optional run_id under --output-root/<dataset>. If omitted, workflow creates timestamp + pipeline summary.")
    parser.add_argument("--task-goal")
    parser.add_argument("--task-type")
    parser.add_argument("--include-qc", action="store_true")
    parser.add_argument("--skip-qc", action="store_true", help="Skip default check-only QC in modeling/splitting workflows.")
    parser.add_argument("--qc-mode", default="check")
    parser.add_argument("--split-ratio")
    parser.add_argument("--confirm-incomplete-split-ratio", action="store_true", help="User confirmed how to complete a malformed ratio such as 6:2:.")
    parser.add_argument("--split-method")
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--n-repeats", type=int)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--preprocess-methods")
    parser.add_argument("--window-length", type=int)
    parser.add_argument("--polyorder", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--poly-degree", type=int)
    parser.add_argument("--als-lambda", type=float)
    parser.add_argument("--als-p", type=float)
    parser.add_argument("--als-iter", type=int)
    parser.add_argument("--band-range")
    parser.add_argument("--remove-band-ranges")
    parser.add_argument("--confirm-baseline", action="store_true")
    parser.add_argument("--confirm-absorbance", action="store_true")
    parser.add_argument("--confirm-band-change", action="store_true")
    parser.add_argument("--confirm-unsplit-preprocess-fit", action="store_true")
    parser.add_argument("--feature-method")
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
    parser.add_argument("--feature-n-runs", type=int)
    parser.add_argument("--feature-sample-ratio", type=float)
    parser.add_argument("--feature-cv", type=int)
    parser.add_argument("--feature-random-state", type=int)
    parser.add_argument("--correlation-method", choices=["pearson", "spearman"])
    parser.add_argument("--interval-mode")
    parser.add_argument("--feature-config")
    parser.add_argument("--auto-confirm-feature-defaults", action="store_true")
    parser.add_argument("--confirm-unsplit-feature-fit", action="store_true")
    parser.add_argument("--models")
    parser.add_argument("--model-config")
    parser.add_argument("--model-n-components", type=int)
    parser.add_argument("--model-embedding-dim", type=int)
    parser.add_argument("--model-epochs", type=int)
    parser.add_argument("--model-batch-size", type=int)
    parser.add_argument("--model-alpha", type=float)
    parser.add_argument("--model-lr", type=float)
    parser.add_argument("--model-kernel")
    parser.add_argument("--model-preprojection")
    parser.add_argument("--model-encoder-type")
    parser.add_argument("--model-metric")
    parser.add_argument("--model-temperature", type=float)
    parser.add_argument("--model-device")
    parser.add_argument("--modeling-mode", choices=["auto", "standard", "repeated_classifier_comparison"], default="auto")
    parser.add_argument("--checkpoint-per-model", action="store_true")
    parser.add_argument("--candidate-model-set-source")
    parser.add_argument("--auto-confirm-model-defaults", action="store_true")
    parser.add_argument("--require-test-confirmation", action="store_true")
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    parser.add_argument("--confirm-confirmatory-test-evaluation", action="store_true")
    parser.add_argument("--reader-sample-orientation", choices=["rows", "columns"])
    parser.add_argument("--reader-label-column")
    parser.add_argument("--reader-target-columns")
    parser.add_argument("--reader-sample-id-column")
    parser.add_argument("--reader-sample-id-column-index", type=int)
    parser.add_argument("--reader-spectral-start-column")
    parser.add_argument("--reader-spectral-end-column")
    parser.add_argument("--reader-band-type")
    parser.add_argument("--reader-band-unit")
    parser.add_argument("--reader-max-auto-columns", "--max-auto-columns", dest="reader_max_auto_columns", type=int, default=10000)
    parser.add_argument("--reader-max-spectral-columns", "--max-spectral-columns", dest="reader_max_spectral_columns", type=int, default=20000)
    parser.add_argument("--reader-wide-table-mode", "--wide-table-mode", dest="reader_wide_table_mode", choices=["auto", "off"], default="auto")
    parser.add_argument("--reader-confirm-wide-table", "--confirm-wide-table", dest="reader_confirm_wide_table", action="store_true")
    parser.add_argument("--reader-confirm-read-plan", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = run_spectral_workflow(
        input_path=_resolve_cli_path(args.input),
        package_dir=_resolve_cli_path(args.package_dir),
        data_contract=_resolve_cli_path(args.data_contract),
        split_contract=_resolve_cli_path(args.split_contract),
        output_dir=_resolve_cli_path(args.output_dir),
        output_root=_resolve_cli_path(args.output_root),
        run_name=args.run_name,
        task_goal=args.task_goal,
        task_type=args.task_type,
        include_qc=args.include_qc,
        skip_qc=args.skip_qc,
        qc_mode=args.qc_mode,
        split_ratio=args.split_ratio,
        confirm_incomplete_split_ratio=args.confirm_incomplete_split_ratio,
        split_method=args.split_method,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        shuffle=not args.no_shuffle,
        preprocess_methods=args.preprocess_methods,
        preprocess_window_length=args.window_length,
        preprocess_polyorder=args.polyorder,
        preprocess_sigma=args.sigma,
        preprocess_poly_degree=args.poly_degree,
        preprocess_als_lambda=args.als_lambda,
        preprocess_als_p=args.als_p,
        preprocess_als_iter=args.als_iter,
        preprocess_band_range=args.band_range,
        preprocess_remove_band_ranges=args.remove_band_ranges,
        confirm_baseline=args.confirm_baseline,
        confirm_absorbance=args.confirm_absorbance,
        confirm_band_change=args.confirm_band_change,
        confirm_unsplit_preprocess_fit=args.confirm_unsplit_preprocess_fit,
        feature_method=args.feature_method,
        feature_n_components=args.n_components,
        feature_explained_variance=args.explained_variance,
        feature_variance_threshold=args.variance_threshold,
        feature_band_min=args.band_min,
        feature_band_max=args.band_max,
        feature_band_indices=args.band_indices,
        feature_names=args.feature_names,
        feature_index_base=args.index_base,
        feature_top_k=args.top_k,
        feature_score_threshold=args.score_threshold,
        feature_n_intervals=args.n_intervals,
        feature_n_runs=args.feature_n_runs,
        feature_sample_ratio=args.feature_sample_ratio,
        feature_cv=args.feature_cv,
        feature_random_state=args.feature_random_state,
        feature_correlation_method=args.correlation_method,
        feature_interval_mode=args.interval_mode,
        feature_config=_resolve_cli_path(args.feature_config),
        auto_confirm_feature_defaults=args.auto_confirm_feature_defaults,
        confirm_unsplit_feature_fit=args.confirm_unsplit_feature_fit,
        models=args.models,
        model_config=_resolve_cli_path(args.model_config),
        model_n_components=args.model_n_components,
        model_embedding_dim=args.model_embedding_dim,
        model_epochs=args.model_epochs,
        model_batch_size=args.model_batch_size,
        model_alpha=args.model_alpha,
        model_lr=args.model_lr,
        model_kernel=args.model_kernel,
        model_preprojection=args.model_preprojection,
        model_encoder_type=args.model_encoder_type,
        model_metric=args.model_metric,
        model_temperature=args.model_temperature,
        model_device=args.model_device,
        modeling_mode=args.modeling_mode,
        checkpoint_per_model=args.checkpoint_per_model,
        candidate_model_set_source=args.candidate_model_set_source,
        auto_confirm_model_defaults=args.auto_confirm_model_defaults,
        require_test_confirmation=args.require_test_confirmation,
        confirm_test_evaluation=args.confirm_test_evaluation,
        confirm_confirmatory_test_evaluation=args.confirm_confirmatory_test_evaluation,
        reader_sample_orientation=args.reader_sample_orientation,
        reader_label_column=args.reader_label_column,
        reader_target_columns=_split_csv_arg(args.reader_target_columns),
        reader_sample_id_column=args.reader_sample_id_column,
        reader_sample_id_column_index=args.reader_sample_id_column_index,
        reader_spectral_start_column=args.reader_spectral_start_column,
        reader_spectral_end_column=args.reader_spectral_end_column,
        reader_band_type=args.reader_band_type,
        reader_band_unit=args.reader_band_unit,
        reader_max_auto_columns=args.reader_max_auto_columns,
        reader_max_spectral_columns=args.reader_max_spectral_columns,
        reader_wide_table_mode=args.reader_wide_table_mode,
        reader_confirm_wide_table=args.reader_confirm_wide_table,
        reader_confirm_read_plan=args.reader_confirm_read_plan,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_cli_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
