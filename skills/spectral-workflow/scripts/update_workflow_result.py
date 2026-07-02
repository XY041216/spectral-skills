"""Update workflow_result.json with stage outputs."""

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

from spectral_core.workflow.state import update_workflow_result
from spectral_core.workflow.run_layout import STAGE_DIRS, RunLayout, relative_stage_outputs, update_runs_index, write_run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update workflow_result.json for a spectral workflow.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--task-goal", required=True)
    parser.add_argument("--stage-output", action="append", default=[], help="Stage output in stage=path form. May be repeated.")
    parser.add_argument("--stage-output-json", help="JSON file containing a {stage: path} object. Safer than stage=path on some shells.")
    parser.add_argument("--stage-output-key", action="append", default=[], help="Stage output key. Pair with --stage-output-path; may be repeated.")
    parser.add_argument("--stage-output-path", action="append", default=[], help="Stage output path. Pair with --stage-output-key; may be repeated.")
    parser.add_argument("--final-output")
    parser.add_argument("--final-output-relative")
    parser.add_argument("--workflow-plan")
    parser.add_argument("--workflow-status", default="ready")
    parser.add_argument("--run-id")
    parser.add_argument("--dataset-name")
    parser.add_argument("--run-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    stage_outputs = _merge_stage_outputs(
        _parse_stage_outputs(args.stage_output),
        _parse_stage_output_json(args.stage_output_json),
        _parse_stage_output_pairs(args.stage_output_key, args.stage_output_path),
    )
    inferred = _infer_run_metadata(
        result_path=Path(args.result),
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        run_dir=args.run_dir,
        output_root=args.output_root,
    )
    stage_outputs_relative = relative_stage_outputs(stage_outputs, inferred["run_dir"]) if inferred.get("run_dir") else None
    response = {
        "ok": True,
        "tool": "update_workflow_result",
        "result": update_workflow_result(
            result_path=args.result,
            task_goal=args.task_goal,
            stage_outputs=stage_outputs,
            stage_outputs_relative=stage_outputs_relative,
            final_output=args.final_output,
            final_output_relative=args.final_output_relative,
            workflow_plan=args.workflow_plan,
            workflow_status=args.workflow_status,
            run_id=inferred.get("run_id"),
            dataset_name=inferred.get("dataset_name"),
            run_dir=inferred.get("run_dir"),
            output_root=inferred.get("output_root"),
        ),
    }
    _update_run_files_if_possible(args, inferred, stage_outputs)
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


def _parse_stage_outputs(values: list[str]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--stage-output must be in stage=path form.")
        stage, path = value.split("=", 1)
        outputs[stage.strip()] = path.strip()
    return outputs


def _parse_stage_output_json(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--stage-output-json must contain a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}


def _parse_stage_output_pairs(keys: list[str], paths: list[str]) -> dict[str, str]:
    if len(keys) != len(paths):
        raise ValueError("--stage-output-key and --stage-output-path must be provided the same number of times.")
    return {key.strip(): path.strip() for key, path in zip(keys, paths)}


def _merge_stage_outputs(*items: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for item in items:
        merged.update(item)
    return merged


def _infer_run_metadata(*, result_path: Path, run_id: str | None, dataset_name: str | None, run_dir: str | None, output_root: str | None) -> dict[str, str | None]:
    inferred_run_dir = Path(run_dir) if run_dir else result_path.parent
    inferred_run_id = run_id or inferred_run_dir.name
    inferred_dataset = dataset_name
    inferred_root = Path(output_root) if output_root else None
    if inferred_dataset is None and inferred_run_dir.parent.name:
        inferred_dataset = inferred_run_dir.parent.name
    if inferred_root is None and inferred_run_dir.parent.parent.name:
        maybe_root = inferred_run_dir.parent.parent
        if maybe_root.name == "spectral_runs":
            inferred_root = maybe_root
    return {
        "run_id": inferred_run_id,
        "dataset_name": inferred_dataset,
        "run_dir": str(inferred_run_dir.resolve()) if inferred_run_dir else None,
        "output_root": str(inferred_root.resolve()) if inferred_root else None,
    }


def _update_run_files_if_possible(args: argparse.Namespace, inferred: dict[str, str | None], stage_outputs: dict[str, str]) -> None:
    if not inferred.get("run_dir"):
        return
    run_dir = Path(str(inferred["run_dir"]))
    output_root = Path(str(inferred["output_root"])) if inferred.get("output_root") else None
    dataset_name = str(inferred.get("dataset_name") or run_dir.parent.name)
    run_id = str(inferred.get("run_id") or run_dir.name)
    layout = RunLayout(
        output_root=output_root,
        dataset_name=dataset_name,
        run_id=run_id,
        run_dir=run_dir,
        stage_dirs={stage: run_dir / dirname for stage, dirname in STAGE_DIRS.items()},
        managed_root=output_root is not None,
    )
    write_run_manifest(layout, task_goal=args.task_goal, status=args.workflow_status, parameters={"final_output": args.final_output})
    if output_root is not None:
        update_runs_index(layout, task_goal=args.task_goal, status=args.workflow_status, stage_outputs=stage_outputs, final_output=args.final_output)


if __name__ == "__main__":
    raise SystemExit(main())
