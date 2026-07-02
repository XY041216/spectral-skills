"""Record a confirmed or pending workflow-stage decision."""

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

from spectral_core.workflow.state import (
    load_decision_file,
    merge_decision_parameters,
    parse_decision_pairs,
    parse_json_object,
    update_workflow_decision,
)


def _stage_explicit_parameters(args: argparse.Namespace) -> dict[str, object]:
    explicit: dict[str, object] = {}
    if args.split_ratio is not None:
        explicit["ratio"] = args.split_ratio
    if args.split_method is not None:
        explicit["method"] = args.split_method
    if args.train_ratio is not None:
        explicit["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        explicit["val_ratio"] = args.val_ratio
    if args.test_ratio is not None:
        explicit["test_ratio"] = args.test_ratio
    if args.n_splits is not None:
        explicit["n_splits"] = args.n_splits
    if args.n_repeats is not None:
        explicit["n_repeats"] = args.n_repeats
    if args.no_shuffle:
        explicit["shuffle"] = False
    if args.random_seed is not None:
        explicit["random_seed"] = args.random_seed
    if args.preprocess_methods is not None:
        explicit["methods"] = args.preprocess_methods
    if args.feature_method is not None:
        explicit["method" if args.stage == "feature" else "feature_method"] = args.feature_method
    if args.models is not None:
        explicit["models"] = args.models
    return explicit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update a stage decision inside workflow_plan.json.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", default="confirmed")
    parser.add_argument("--decision-source", default="user_confirmed_recommendation")
    parser.add_argument("--question")
    parser.add_argument("--recommended-option")
    parser.add_argument("--user-selected-option")
    parser.add_argument("--parameters-json")
    parser.add_argument("--decision", action="append", help="PowerShell-friendly key=value parameter. Repeat for multiple decisions.")
    parser.add_argument("--decision-file", help="Path to a JSON object with decision parameters.")
    parser.add_argument("--split-ratio")
    parser.add_argument("--split-method")
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--n-repeats", type=int)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--preprocess-methods")
    parser.add_argument("--feature-method")
    parser.add_argument("--models")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    explicit = _stage_explicit_parameters(args)
    parameters = merge_decision_parameters(
        parse_json_object(args.parameters_json),
        load_decision_file(args.decision_file),
        parse_decision_pairs(args.decision),
        explicit,
    )
    response = {
        "ok": True,
        "tool": "update_workflow_decision",
        "result": update_workflow_decision(
            plan_path=args.plan,
            stage=args.stage,
            status=args.status,
            decision_source=args.decision_source,
            parameters=parameters,
            question=args.question,
            recommended_option=args.recommended_option,
            user_selected_option=args.user_selected_option,
        ),
    }
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
