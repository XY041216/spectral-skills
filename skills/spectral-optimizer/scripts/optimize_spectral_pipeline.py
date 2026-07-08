"""CLI entry for spectral optimizer planning and recommendation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_runtime_root() -> Path:
    here = Path(__file__).resolve()
    candidates = list(here.parents) + [parent / "spectral-core" for parent in here.parents]
    for candidate in candidates:
        if (candidate / "spectral_core" / "__init__.py").is_file():
            return candidate
    return here.parents[3]


ROOT = _find_runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.optimizer.workflow import optimize_spectral_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan leakage-safe spectral optimization candidates without using test data for selection.")
    parser.add_argument("--mode", choices=["recommend_from_profile", "tune_method", "compare_step", "optimize_pipeline"], default="recommend_from_profile")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-profile")
    parser.add_argument("--package-dir")
    parser.add_argument("--split-contract")
    parser.add_argument("--task-type", choices=["classification", "regression"])
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-features", type=int)
    parser.add_argument("--n-classes", type=int)
    parser.add_argument("--class-balance")
    parser.add_argument("--has-validation", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--has-test", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--target-step", choices=["preprocess", "feature", "modeling"])
    parser.add_argument("--method")
    parser.add_argument("--fixed-preprocess-methods", help="Fixed upstream preprocess methods for compare_step feature, e.g. snv. Optimizer executor prepares this after confirmation.")
    parser.add_argument("--fixed-feature-contract", help="Fixed upstream feature_contract.json for compare_step modeling; optimizer uses it directly without Agent-created candidate_space files.")
    parser.add_argument("--preprocess-candidates", help="Comma-separated preprocess methods. For optimize_pipeline these append to the regular preprocess axis; for compare_step preprocess they define the compared methods.")
    parser.add_argument("--feature-candidates", help="Comma-separated feature methods. For optimize_pipeline these append to the regular feature axis and cross with preprocess/model axes, e.g. cls_former_embedding.")
    parser.add_argument("--model-candidates", help="Comma-separated model methods. For optimize_pipeline these append to the regular modeling axis and cross with preprocess/feature axes, e.g. cls_former_classifier,proto_spectral_classifier.")
    parser.add_argument("--validator-model", help="Validator model for preprocess/feature comparison, e.g. svm. This avoids Agent-created candidate_space JSON.")
    parser.add_argument("--validator-param", action="append", help="Locked validator parameter key=value. Repeatable, e.g. --validator-param C=1.0 --validator-param gamma=scale.")
    parser.add_argument("--model-param-grid", action="append", help="Model parameter grid entry model.param=value. Repeatable; use | for multiple values, e.g. svm.C=1|10.")
    parser.add_argument("--comparison-depth", choices=["quick", "regular", "recommended", "extended", "deep"], default="regular", help="Feature comparison budget: quick, regular, extended, or explicitly confirmed deep search. 'recommended' is a backward-compatible alias for regular.")
    parser.add_argument("--candidate-space")
    parser.add_argument("--trial-results")
    parser.add_argument("--execute-trials", action="store_true")
    parser.add_argument("--trial-inputs")
    parser.add_argument("--test-access-log")
    parser.add_argument("--selection-metric")
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--confirm-comparison-design", action="store_true", help="User confirmed compare/tune candidates, fixed stages, validator model, metric, and budget.")
    parser.add_argument("--confirm-parameter-grid", action="store_true", help="User confirmed all parameter grids that will be expanded for compare/tune execution.")
    parser.add_argument("--confirm-candidate-space", action="store_true", help="User confirmed optimize_pipeline candidate stages and compact/extended search policy.")
    parser.add_argument("--validator-model-source", help="Audit source for the validator model, for example inherited_from_previous_user_confirmed_comparison.")
    parser.add_argument("--previous-confirmation-stage", help="Previous stage whose user confirmation selected the inherited validator model.")
    parser.add_argument("--preview-only", action="store_true", help="Return a no-side-effect confirmation card on stdout; do not create output files or directories.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    has_validation = None
    if args.has_validation or args.no_validation:
        has_validation = args.has_validation and not args.no_validation
    has_test = None
    if args.has_test or args.no_test:
        has_test = args.has_test and not args.no_test
    response = optimize_spectral_pipeline(
        mode=args.mode,
        output_dir=_resolve_cli_path(args.output_dir),
        data_profile=_resolve_cli_path(args.data_profile),
        package_dir=_resolve_cli_path(args.package_dir),
        split_contract=_resolve_cli_path(args.split_contract),
        task_type=args.task_type,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        class_balance=args.class_balance,
        has_validation=has_validation,
        has_test=has_test,
        target_step=args.target_step,
        method=args.method,
        fixed_preprocess_methods=args.fixed_preprocess_methods,
        fixed_feature_contract=_resolve_cli_path(args.fixed_feature_contract),
        preprocess_candidates=args.preprocess_candidates,
        feature_candidates=args.feature_candidates,
        model_candidates=args.model_candidates,
        validator_model=args.validator_model,
        validator_params=args.validator_param,
        model_param_grid=args.model_param_grid,
        comparison_depth=args.comparison_depth,
        candidate_space=_resolve_cli_path(args.candidate_space),
        trial_results=_resolve_cli_path(args.trial_results),
        execute_trials=args.execute_trials,
        trial_inputs=_resolve_cli_path(args.trial_inputs),
        test_access_log=_resolve_cli_path(args.test_access_log),
        selection_metric=args.selection_metric,
        max_trials=args.max_trials,
        confirm_budget=args.confirm_budget,
        confirm_comparison_design=args.confirm_comparison_design,
        confirm_parameter_grid=args.confirm_parameter_grid,
        confirm_candidate_space=args.confirm_candidate_space,
        validator_model_source=args.validator_model_source,
        previous_confirmation_stage=args.previous_confirmation_stage,
        preview_only=args.preview_only,
        random_seed=args.random_seed,
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0 if response.get("ok") else 1


def _resolve_cli_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
