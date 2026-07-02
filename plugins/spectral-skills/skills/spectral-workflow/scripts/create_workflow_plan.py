"""Create a structured spectral workflow plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_core.workflow.state import create_workflow_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create workflow_plan.json for a spectral workflow.")
    parser.add_argument("--output-dir", default="workflow_output")
    parser.add_argument("--task-goal")
    parser.add_argument("--input")
    parser.add_argument("--package-dir")
    parser.add_argument("--data-contract")
    parser.add_argument("--split-contract")
    parser.add_argument("--include-qc", action="store_true")
    parser.add_argument("--qc-mode", default="check")
    parser.add_argument("--split-ratio")
    parser.add_argument("--split-method")
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--n-repeats", type=int)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--preprocess-methods")
    parser.add_argument("--feature-method")
    parser.add_argument("--models")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    response = {
        "ok": True,
        "tool": "create_workflow_plan",
        "result": create_workflow_plan(
            output_dir=args.output_dir,
            task_goal=args.task_goal,
            input_path=args.input,
            package_dir=args.package_dir,
            data_contract=args.data_contract,
            split_contract=args.split_contract,
            include_qc=args.include_qc,
            qc_mode=args.qc_mode,
            split_ratio=args.split_ratio,
            split_method=args.split_method,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            shuffle=not args.no_shuffle,
            preprocess_methods=args.preprocess_methods,
            feature_method=args.feature_method,
            models=args.models,
            random_seed=args.random_seed,
        ),
    }
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
