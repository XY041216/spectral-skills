"""CLI entry for leakage-safe modeling of split standard spectral packages."""

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

from spectral_core.modeling.workflow import model_spectral_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate spectral models with train/val/test leakage controls.")
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--split-contract")
    parser.add_argument("--feature-contract", help="Fold/repeat-wise feature_contract.json to model directly.")
    parser.add_argument("--preprocess-contract", help="Fold/repeat-wise preprocess_contract.json to model directly.")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=["standard", "repeated_classifier_comparison"], default="standard", help="standard trains/evaluates one selected model; repeated_classifier_comparison evaluates each classifier on every fold/repeat without internal model selection.")
    parser.add_argument("--task-type", choices=["classification", "regression", "multi_target_regression"])
    parser.add_argument("--models", help="Comma-separated model list.")
    parser.add_argument("--model-config", help="JSON model configuration file.")
    parser.add_argument("--best-pipeline", help="optimizer best_pipeline.json to reproduce exactly.")
    parser.add_argument("--lock-best-pipeline-params", action="store_true", help="Use exactly the model and params from --best-pipeline.")
    parser.add_argument("--disable-model-selection", "--no-param-search", dest="disable_model_selection", action="store_true", help="Disable internal hyperparameter search; use supplied model params exactly.")
    parser.add_argument("--auto-confirm-model-defaults", action="store_true")
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
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--confirm-no-test", action="store_true")
    parser.add_argument("--require-test-confirmation", action="store_true", help="Block final holdout test evaluation unless --confirm-test-evaluation is also supplied.")
    parser.add_argument("--confirm-test-evaluation", action="store_true", help="User explicitly confirmed this is the final/confirmatory test evaluation.")
    parser.add_argument("--confirm-confirmatory-test-evaluation", action="store_true", help="User explicitly confirmed another test evaluation after prior test access.")
    parser.add_argument("--evaluation-mode", choices=["final", "validation_only"], default="final")
    parser.add_argument("--skip-test-evaluation", action="store_true", help="Alias for --evaluation-mode validation_only.")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-per-model", action="store_true", help="For repeated_classifier_comparison, write each model's partial repeated result as soon as it completes.")
    parser.add_argument("--candidate-model-set-source", help="Human-readable source of the confirmed classifier set, such as regular-fast, regular-full, spectral-modeling, or user_custom.")
    parser.add_argument("--confirm-gated-feature-modeling", action="store_true", help="Confirm modeling with a visualization-first/manifold feature contract such as UMAP, Isomap, or LLE. This never enables t-SNE modeling.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = model_spectral_package(
        package_dir=args.package_dir,
        split_contract=args.split_contract,
        feature_contract=args.feature_contract,
        preprocess_contract=args.preprocess_contract,
        output_dir=args.output_dir,
        task_type=args.task_type,
        models=args.models,
        model_config=args.model_config,
        best_pipeline=args.best_pipeline,
        lock_best_pipeline_params=args.lock_best_pipeline_params,
        disable_model_selection=args.disable_model_selection,
        model_parameters={
            "n_components": args.model_n_components,
            "embedding_dim": args.model_embedding_dim,
            "epochs": args.model_epochs,
            "batch_size": args.model_batch_size,
            "alpha": args.model_alpha,
            "lr": args.model_lr,
            "kernel": args.model_kernel,
            "preprojection": args.model_preprojection,
            "encoder_type": args.model_encoder_type,
            "metric": args.model_metric,
            "temperature": args.model_temperature,
            "device": args.model_device,
        },
        auto_confirm_model_defaults=args.auto_confirm_model_defaults,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
        confirm_no_test=args.confirm_no_test,
        require_test_confirmation=args.require_test_confirmation,
        confirm_test_evaluation=args.confirm_test_evaluation,
        confirm_confirmatory_test_evaluation=args.confirm_confirmatory_test_evaluation,
        evaluation_mode="validation_only" if args.skip_test_evaluation else args.evaluation_mode,
        save_model=not args.no_save_model,
        overwrite=args.overwrite,
        backend="script",
        mode=args.mode,
        checkpoint_per_model=args.checkpoint_per_model,
        candidate_model_set_source=args.candidate_model_set_source,
        confirm_gated_feature_modeling=args.confirm_gated_feature_modeling,
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
