"""Run-root layout helpers for spectral workflow outputs."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file, write_json_file


STAGE_DIRS = {
    "reader": "reader_package",
    "qc": "qc_output",
    "splitter": "split_output",
    "preprocess": "preprocess_output",
    "feature": "feature_output",
    "modeling": "model_output",
    "report": "report_output",
    "logs": "logs",
}


@dataclass(frozen=True)
class RunLayout:
    output_root: Path | None
    dataset_name: str
    run_id: str
    run_dir: Path
    stage_dirs: dict[str, Path]
    managed_root: bool


def create_run_layout(
    *,
    output_root: str | Path | None,
    output_dir: str | Path | None,
    run_name: str | None,
    input_path: str | Path | None,
    package_dir: str | Path | None,
    data_contract: str | Path | None,
    split_method: str | None,
    split_ratio: str | None,
    n_splits: int | None,
    n_repeats: int | None,
    preprocess_methods: str | list[str] | None,
    feature_method: str | None,
    feature_n_components: int | None,
    models: str | list[str] | None,
) -> RunLayout:
    dataset_name = dataset_stem(input_path=input_path, package_dir=package_dir, data_contract=data_contract)
    if output_root is not None:
        root = Path(output_root)
        run_id = sanitize_run_id(
            run_name
            or make_run_id(
                split_method=split_method,
                split_ratio=split_ratio,
                n_splits=n_splits,
                n_repeats=n_repeats,
                preprocess_methods=preprocess_methods,
                feature_method=feature_method,
                feature_n_components=feature_n_components,
                models=models,
            )
        )
        run_dir = root / dataset_name / run_id
        managed = True
    else:
        run_dir = Path(output_dir or "workflow_output")
        root = None
        run_id = sanitize_run_id(run_name or run_dir.name)
        managed = False
    stage_dirs = {stage: run_dir / name for stage, name in STAGE_DIRS.items()}
    return RunLayout(output_root=root, dataset_name=dataset_name, run_id=run_id, run_dir=run_dir, stage_dirs=stage_dirs, managed_root=managed)


def ensure_run_dirs(layout: RunLayout) -> None:
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    for path in layout.stage_dirs.values():
        path.mkdir(parents=True, exist_ok=True)


def make_run_id(
    *,
    split_method: str | None,
    split_ratio: str | None,
    n_splits: int | None,
    n_repeats: int | None,
    preprocess_methods: str | list[str] | None,
    feature_method: str | None,
    feature_n_components: int | None,
    models: str | list[str] | None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [_split_label(split_method, split_ratio, n_splits, n_repeats)]
    parts.extend(_method_list(preprocess_methods))
    if feature_method and str(feature_method).strip().lower() not in {"none", "skip", "no_feature"}:
        feature = str(feature_method).strip().lower().replace("-", "_")
        if feature == "pca" and feature_n_components:
            feature = f"pca{feature_n_components}"
        parts.append(feature)
    parts.extend(_method_list(models))
    summary = "_".join(part for part in parts if part and part != "none")
    return sanitize_run_id(f"{timestamp}_{summary or 'workflow'}")


def dataset_stem(*, input_path: str | Path | None, package_dir: str | Path | None, data_contract: str | Path | None) -> str:
    if input_path is not None:
        path = Path(input_path)
        if path.name == "data_contract.json":
            return _dataset_from_contract(path)
        if path.suffix:
            return sanitize_name(path.stem)
        return sanitize_name(path.name)
    if data_contract is not None:
        return _dataset_from_contract(Path(data_contract))
    if package_dir is not None:
        contract = Path(package_dir) / "data_contract.json"
        if contract.exists():
            return _dataset_from_contract(contract)
        return sanitize_name(_strip_package_suffix(Path(package_dir).name))
    return "dataset"


def write_run_manifest(layout: RunLayout, *, task_goal: str | None, parameters: dict[str, Any], status: str) -> Path:
    path = layout.run_dir / "run_manifest.json"
    existing = load_json_file(path) if path.exists() else {}
    existing_parameters = existing.get("parameters") if isinstance(existing.get("parameters"), dict) else {}
    merged_parameters = dict(existing_parameters)
    merged_parameters.update({key: value for key, value in parameters.items() if value is not None})
    payload = {
        "run_id": layout.run_id,
        "dataset_name": layout.dataset_name,
        "run_dir": str(layout.run_dir.resolve()),
        "output_root": str(layout.output_root.resolve()) if layout.output_root else None,
        "task_goal": task_goal,
        "parameters": merged_parameters,
        "status": status,
        "created_at": existing.get("created_at") or datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    write_json_file(path, payload, ensure_ascii=False)
    return path


def update_runs_index(layout: RunLayout, *, task_goal: str, status: str, stage_outputs: dict[str, str], final_output: str | None) -> None:
    if not layout.managed_root or layout.output_root is None:
        return
    dataset_root = layout.output_root / layout.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    index_path = dataset_root / "runs_index.csv"
    row = {
        "run_id": layout.run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task_goal,
        "split_method": _stage_method(layout.run_dir / "split_output" / "split_contract.json"),
        "preprocess": _stage_methods(layout.run_dir / "preprocess_output" / "preprocess_contract.json", "methods"),
        "feature": _stage_methods(layout.run_dir / "feature_output" / "feature_contract.json", "feature_method"),
        "model": _model_name(final_output),
        "status": status,
        "final_metric": _final_metric(final_output),
        "run_dir": str(layout.run_dir.resolve()),
    }
    fieldnames = ["run_id", "created_at", "task", "split_method", "preprocess", "feature", "model", "status", "final_metric", "run_dir"]
    rows = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [item for item in csv.DictReader(handle) if item.get("run_id") != layout.run_id]
    rows.append(row)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (dataset_root / "latest.txt").write_text(layout.run_id, encoding="utf-8")


def relative_stage_outputs(stage_outputs: dict[str, str], run_dir: str | Path) -> dict[str, str]:
    root = Path(run_dir).resolve()
    output: dict[str, str] = {}
    for key, value in stage_outputs.items():
        if _is_special_output(value):
            output[key] = value
            continue
        try:
            output[key] = str(Path(value).resolve().relative_to(root)).replace("\\", "/")
        except Exception:
            output[key] = value
    return output


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "dataset"


def sanitize_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or make_run_id(split_method=None, split_ratio=None, n_splits=None, n_repeats=None, preprocess_methods=None, feature_method=None, feature_n_components=None, models=None)


def _split_label(method: str | None, ratio: str | None, n_splits: int | None, n_repeats: int | None) -> str:
    normalized = str(method or "holdout").lower().replace("-", "_")
    if normalized in {"kfold", "stratified_kfold"}:
        return f"kfold{n_splits or 5}"
    if normalized == "leave_one_out":
        return "loocv"
    if normalized in {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"}:
        return f"mccv{n_repeats or 100}"
    if ratio:
        return f"holdout{re.sub(r'[^0-9]', '', str(ratio))}"
    if normalized in {"random", "stratified"}:
        return "holdout"
    return normalized


def _method_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return [sanitize_name(item.lower().replace("-", "_")) for item in items if item.lower() not in {"none", "skip", "no", "no_feature", "no_preprocess"}]


def _dataset_from_contract(path: Path) -> str:
    try:
        contract = load_json_file(path)
    except Exception:
        return sanitize_name(_strip_package_suffix(path.parent.name))
    for key in ["source_file", "input_path", "dataset_name", "dataset_id", "file_name", "original_file", "input_file"]:
        if contract.get(key):
            value = str(contract[key])
            return sanitize_name(Path(value).stem if Path(value).suffix else value)
    source = contract.get("source")
    if isinstance(source, dict) and source.get("file"):
        return sanitize_name(Path(str(source["file"])).stem)
    if isinstance(source, dict) and source.get("input"):
        return sanitize_name(Path(str(source["input"])).stem)
    files = contract.get("files")
    if isinstance(files, dict):
        for key in ["source", "input", "raw", "original"]:
            if files.get(key):
                return sanitize_name(Path(str(files[key])).stem)
    return sanitize_name(_strip_package_suffix(path.parent.name))


def _strip_package_suffix(name: str) -> str:
    for suffix in ["_reader_package", "_reader_output", "_package", "_workflow", "_output"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _stage_method(path: Path) -> str:
    try:
        return str(load_json_file(path).get("method") or "")
    except Exception:
        return ""


def _stage_methods(path: Path, key: str) -> str:
    try:
        payload = load_json_file(path)
    except Exception:
        return "none"
    value = payload.get(key)
    if value is None and key == "methods":
        value = payload.get("executed_methods")
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "none")


def _model_name(final_output: str | None) -> str:
    if not final_output:
        return ""
    try:
        payload = load_json_file(Path(final_output))
    except Exception:
        return ""
    for key in ["model_type", "selected_model"]:
        if payload.get(key):
            return str(payload[key])
    candidates = payload.get("candidate_models")
    if isinstance(candidates, list) and candidates:
        return ",".join(str(item) for item in candidates)
    return ""


def _final_metric(final_output: str | None) -> str:
    if not final_output:
        return ""
    path = Path(final_output)
    candidates = [path.parent / "metric_summary.json", path.parent / "metrics.json", path]
    for candidate in candidates:
        try:
            payload = load_json_file(candidate)
        except Exception:
            continue
        value = _metric_from_payload(payload)
        if value != "":
            return value
    return ""


def _metric_from_payload(payload: dict[str, Any]) -> str:
    for container_key in ["metrics_mean", "test_metrics", "test", "metric_summary"]:
        container = payload.get(container_key)
        if isinstance(container, dict):
            for metric in ["accuracy", "macro_f1", "r2", "rmse"]:
                if metric in container:
                    return str(container[metric])
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        test = metrics.get("test")
        if isinstance(test, dict):
            for metric in ["accuracy", "macro_f1", "r2", "rmse"]:
                if metric in test:
                    return str(test[metric])
    return ""


def _is_special_output(value: str) -> bool:
    return str(value).startswith("reused_from:") or str(value).startswith("skipped_")
