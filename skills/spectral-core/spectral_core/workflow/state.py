"""Structured workflow state helpers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spectral_core.reader.io_utils import load_json_file, write_json_file
from spectral_core.splitter.ratios import resolve_ratios


WORKFLOW_PLAN_VERSION = "0.2.0"
STAGE_ORDER = ["reader", "qc", "splitter", "preprocess", "feature", "modeling"]
WORKFLOW_LOCK_TIMEOUT_SECONDS = 30.0
WORKFLOW_LOCK_POLL_SECONDS = 0.05


def normalize_workflow_goal(task_goal: str | None) -> str | None:
    if task_goal is None or not str(task_goal).strip():
        return None
    text = str(task_goal).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "read_only": "read",
        "quality_check": "qc",
        "check_quality": "qc",
        "splitter": "split",
        "splitting": "split",
        "prepare_optimizer": "prepare_for_optimizer",
        "optimizer_prepare": "prepare_for_optimizer",
        "prepare_for_optimization": "prepare_for_optimizer",
        "compare_preprocessing": "compare_preprocess",
        "classify": "classification",
        "classifier": "classification",
        "classification_baseline": "classification",
        "baseline_classification": "classification",
        "regress": "regression",
        "regression_baseline": "regression",
        "baseline_regression": "regression",
        "model": "modeling",
    }
    return aliases.get(text, text)


def create_workflow_plan(
    *,
    output_dir: str | Path,
    task_goal: str | None,
    input_path: str | Path | None = None,
    package_dir: str | Path | None = None,
    data_contract: str | Path | None = None,
    split_contract: str | Path | None = None,
    include_qc: bool = False,
    qc_mode: str = "check",
    split_ratio: str | None = None,
    split_method: str | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    shuffle: bool = True,
    preprocess_methods: str | list[str] | None = None,
    preprocess_parameters: dict[str, Any] | None = None,
    feature_method: str | None = None,
    feature_parameters: dict[str, Any] | None = None,
    models: str | list[str] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    task_goal = normalize_workflow_goal(task_goal)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    modeling_goal = task_goal in {"classification", "regression", "modeling"}
    inferred_feature_none = modeling_goal and feature_method is None and models is not None and preprocess_methods is not None
    stages: list[dict[str, Any]] = []
    has_package = package_dir is not None or data_contract is not None
    stages.append(
        _stage_plan(
            "reader",
            "skip" if has_package else "execute" if input_path is not None else "pending_user_decision",
            skip_reason="standard package already supplied" if has_package else None,
            required_fields=[] if input_path is not None or has_package else ["input_path"],
        )
    )
    stages.append(
        _stage_plan(
            "qc",
            "execute" if include_qc or task_goal == "qc" else "skip",
            parameters={"mode": qc_mode},
            skip_reason="QC was not requested" if not include_qc and task_goal != "qc" else None,
        )
    )
    if task_goal in {"split", "preprocess", "feature", "classification", "regression", "modeling"}:
        split_parameters = _split_parameters(
            split_method=split_method,
            split_ratio=split_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_splits=n_splits,
            n_repeats=n_repeats,
            shuffle=shuffle,
            random_seed=random_seed,
        )
        missing = _missing_split_fields(split_contract=split_contract, split_method=split_method, split_ratio=split_ratio)
        stages.append(
            _stage_plan(
                "splitter",
                "user_specified" if split_contract is not None else "pending_user_decision" if missing else "execute",
                parameters=split_parameters,
                required_fields=missing,
                decision_source="user_specified" if split_contract is not None or split_method is not None else None,
            )
        )
    else:
        stages.append(_stage_plan("splitter", "skip", skip_reason="No train/validation/test stage is needed for the requested goal."))
    if task_goal in {"preprocess", "classification", "regression", "modeling"}:
        if _is_none_selection(preprocess_methods):
            stages.append(
                _stage_plan(
                    "preprocess",
                    "skip",
                    parameters={"methods": "none"},
                    skip_reason="User explicitly selected no preprocessing.",
                    decision_source="user_specified",
                )
            )
        else:
            stages.append(
                _stage_plan(
                    "preprocess",
                    "pending_user_decision" if preprocess_methods is None else "execute",
                    parameters=_merge_params({"methods": preprocess_methods}, preprocess_parameters),
                    required_fields=["preprocess_methods"] if preprocess_methods is None else [],
                    decision_source="user_specified" if preprocess_methods is not None else None,
                )
            )
    else:
        stages.append(_stage_plan("preprocess", "skip", skip_reason="Preprocessing was not requested."))
    if task_goal in {"feature", "classification", "regression", "modeling"}:
        if _is_none_selection(feature_method) or inferred_feature_none:
            stages.append(
                _stage_plan(
                    "feature",
                    "skip",
                    parameters={"method": "none"},
                    skip_reason="No feature method was requested; modeling follows preprocessing directly." if inferred_feature_none else "User explicitly selected no feature engineering.",
                    decision_source="inferred_from_user_pipeline" if inferred_feature_none else "user_specified",
                )
            )
        else:
            stages.append(
                _stage_plan(
                    "feature",
                    "pending_user_decision" if feature_method is None else "execute",
                    parameters=_merge_params({"method": feature_method}, feature_parameters),
                    required_fields=["feature_method"] if feature_method is None else [],
                    decision_source="user_specified" if feature_method is not None else None,
                )
            )
    else:
        stages.append(_stage_plan("feature", "skip", skip_reason="Feature engineering was not requested."))
    if modeling_goal:
        stages.append(_stage_plan("modeling", "pending_user_decision" if models is None else "execute", parameters={"models": models}, required_fields=["models"] if models is None else []))
    else:
        stages.append(_stage_plan("modeling", "skip", skip_reason="Modeling was not requested."))

    payload = {
        "workflow_plan_version": WORKFLOW_PLAN_VERSION,
        "task_goal": task_goal,
        "status": "pending_user_decision" if any(stage["status"] == "pending_user_decision" for stage in stages) else "ready_to_execute",
        "contract_chain": ["data_contract.json", "qc_result.json", "split_contract.json", "preprocess data_contract.json", "feature data_contract.json", "modeling_contract.json"],
        "stages": stages,
        "confirmation_log": [],
        "execution": {"created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()},
    }
    path = root / "workflow_plan.json"
    with _workflow_file_lock(path):
        write_json_file(path, payload, ensure_ascii=False)
    return {"status": payload["status"], "workflow_plan": str(path), "plan": payload}


def update_workflow_decision(
    *,
    plan_path: str | Path,
    stage: str,
    decision_source: str,
    status: str = "confirmed",
    parameters: dict[str, Any] | None = None,
    question: str | None = None,
    recommended_option: Any = None,
    user_selected_option: Any = None,
) -> dict[str, Any]:
    path = Path(plan_path)
    with _workflow_file_lock(path):
        payload = load_json_file(path)
        now = datetime.now(timezone.utc).isoformat()
        stage_item = _find_stage(payload, stage)
        existing_parameters = dict(stage_item.get("parameters") or {})
        explicit_parameters = dict(parameters or {})
        if parameters:
            merged = dict(existing_parameters)
            merged.update(parameters)
            stage_item["parameters"] = merged
        final_parameters = dict(stage_item.get("parameters") or {})
        parameter_decisions = _parameter_decisions(
            stage=stage,
            explicit_parameters=explicit_parameters,
            final_parameters=final_parameters,
            decision_source=decision_source,
        )
        stage_item["status"] = "execute" if status == "confirmed" else status
        stage_item["confirmation"] = {
            "required": True,
            "status": status,
            "decision_source": decision_source,
            "question": question,
            "recommended_option": recommended_option,
            "user_selected_option": user_selected_option,
            "required_fields": [],
            "parameter_decisions": parameter_decisions,
            "confirmed_at": now if status == "confirmed" else None,
        }
        log = payload.setdefault("confirmation_log", [])
        log.append(
            {
                "stage": stage,
                "status": status,
                "decision_source": decision_source,
                "question": question,
                "recommended_option": recommended_option,
                "user_selected_option": user_selected_option,
                "parameters": parameters or {},
                "parameter_decisions": parameter_decisions,
                "timestamp": now,
            }
        )
        payload["status"] = "pending_user_decision" if any(item.get("status") == "pending_user_decision" for item in payload.get("stages", [])) else "ready_to_execute"
        payload.setdefault("execution", {})["updated_at"] = now
        write_json_file(path, payload, ensure_ascii=False)
    return {"status": payload["status"], "workflow_plan": str(path), "stage": stage, "plan": payload}


def update_workflow_result(
    *,
    result_path: str | Path,
    task_goal: str,
    stage_outputs: dict[str, str],
    stage_outputs_relative: dict[str, str] | None = None,
    final_output: str | None = None,
    final_output_relative: str | None = None,
    workflow_plan: str | None = None,
    workflow_status: str = "ready",
    confirmation_required: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    dataset_name: str | None = None,
    run_dir: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _workflow_file_lock(path):
        existing = load_json_file(path) if path.exists() else {}
        outputs = dict(existing.get("stage_outputs") or {})
        outputs.update(stage_outputs)
        payload = {
            "workflow_status": workflow_status,
            "task_goal": task_goal,
            "run_id": run_id or existing.get("run_id"),
            "dataset_name": dataset_name or existing.get("dataset_name"),
            "run_dir": run_dir or existing.get("run_dir"),
            "output_root": output_root or existing.get("output_root"),
            "stage_outputs": outputs,
            "stage_outputs_relative": stage_outputs_relative or existing.get("stage_outputs_relative") or {},
            "final_output": final_output or existing.get("final_output") or (list(outputs.values())[-1] if outputs else None),
            "final_output_relative": final_output_relative or existing.get("final_output_relative"),
            "workflow_plan": workflow_plan or existing.get("workflow_plan"),
            "handoff_ready": workflow_status == "ready",
            "confirmation_required": confirmation_required if confirmation_required is not None else existing.get("confirmation_required", []),
            "warnings": warnings if warnings is not None else existing.get("warnings", []),
            "execution": {"updated_at": datetime.now(timezone.utc).isoformat()},
        }
        write_json_file(path, payload, ensure_ascii=False)
    result = dict(payload)
    result["workflow_result"] = str(path)
    return result


@contextmanager
def _workflow_file_lock(path: str | Path, *, timeout: float = WORKFLOW_LOCK_TIMEOUT_SECONDS):
    """Serialize read-modify-write updates for workflow JSON files.

    The lock is an adjacent exclusive-create file, which works on local
    Windows and POSIX filesystems without optional dependencies.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f"{target.name}.lock")
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}\n".encode("utf-8"))
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for workflow JSON lock: {lock_path}")
            time.sleep(WORKFLOW_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _stage_plan(
    stage: str,
    status: str,
    *,
    parameters: dict[str, Any] | None = None,
    required_fields: list[str] | None = None,
    skip_reason: str | None = None,
    decision_source: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "parameters": parameters or {},
        "confirmation": {
            "required": status == "pending_user_decision",
            "status": "pending_user_decision" if status == "pending_user_decision" else "not_required",
            "required_fields": required_fields or [],
        },
    }
    if skip_reason:
        payload["skip_reason"] = skip_reason
    if decision_source:
        payload["decision_source"] = decision_source
    return payload


def _find_stage(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    for item in payload.get("stages", []):
        if item.get("stage") == stage:
            return item
    raise ValueError(f"workflow_plan.json has no stage named {stage!r}")


def parse_json_object(value: str | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def parse_decision_pairs(values: list[str] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Decision must use key=value form: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Decision key is empty: {raw!r}")
        output[_normalize_decision_key(key)] = _parse_scalar(value)
    return output


def load_decision_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_json_file(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("--decision-file must contain a JSON object.")
    return {_normalize_decision_key(str(key)): value for key, value in payload.items()}


def merge_decision_parameters(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        merged.update(item)
    return merged


def _parameter_decisions(
    *,
    stage: str,
    explicit_parameters: dict[str, Any],
    final_parameters: dict[str, Any],
    decision_source: str,
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for key, value in explicit_parameters.items():
        decisions[key] = {"value": value, "decision_source": decision_source}
    if stage == "splitter":
        seed_value = final_parameters.get("random_seed")
        if seed_value is not None:
            if "random_seed" in explicit_parameters:
                seed_source = (
                    "recommended_default_confirmed_with_split"
                    if decision_source in {"user_confirmed_recommendation", "user_specified"}
                    else decision_source
                )
            else:
                seed_source = "recommended_default_used"
            decisions["random_seed"] = {"value": seed_value, "decision_source": seed_source}
    return decisions


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _normalize_decision_key(key: str) -> str:
    normalized = key.strip().replace("-", "_")
    aliases = {
        "split_ratio": "ratio",
        "split_method": "method",
        "preprocess_methods": "methods",
        "preprocess_method": "methods",
        "feature": "method",
        "feature_method": "method",
        "model": "models",
    }
    return aliases.get(normalized, normalized)


def _is_none_selection(value: str | list[str] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        parts = [str(item).strip().lower() for item in value if str(item).strip()]
    return bool(parts) and all(item in {"none", "skip", "no", "no_preprocess", "no_feature"} for item in parts)


def _merge_params(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if extra:
        merged.update({key: value for key, value in extra.items() if value is not None and value is not False and value != ""})
    return merged


def _split_type_for_method(method: str | None) -> str | None:
    if method is None:
        return None
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"kfold", "stratified_kfold", "leave_one_out"}:
        return "cross_validation"
    if normalized in {"monte_carlo_cv", "repeated_random_split", "stratified_monte_carlo_cv"}:
        return "repeated_holdout"
    if normalized == "predefined_split":
        return "predefined"
    return "holdout"


def _split_parameters(
    *,
    split_method: str | None,
    split_ratio: str | None,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    n_splits: int | None,
    n_repeats: int | None,
    shuffle: bool,
    random_seed: int,
) -> dict[str, Any]:
    split_type = _split_type_for_method(split_method)
    parameters: dict[str, Any] = {"method": split_method, "split_type": split_type, "random_seed": random_seed}
    if split_type in {None, "holdout"}:
        parameters.update({"ratio": split_ratio, "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio})
    elif split_type == "cross_validation":
        parameters.update({"n_splits": n_splits or 5, "shuffle": shuffle})
    elif split_type == "repeated_holdout":
        parsed_ratio = _parse_split_ratio_for_plan(split_ratio)
        parameters.update(
            {
                "ratio": split_ratio,
                "n_repeats": n_repeats or 100,
                "train_ratio": _choose_ratio_value(train_ratio, parsed_ratio, "train", 0.7),
                "val_ratio": _choose_ratio_value(val_ratio, parsed_ratio, "val", None),
                "test_ratio": _choose_ratio_value(test_ratio, parsed_ratio, "test", 0.3),
            }
        )
    elif split_type == "predefined":
        parameters.update({"ratio": split_ratio})
    return parameters


def _parse_split_ratio_for_plan(split_ratio: str | None) -> dict[str, float] | None:
    if not split_ratio:
        return None
    try:
        return resolve_ratios(ratio=split_ratio)
    except Exception:
        return None


def _choose_ratio_value(explicit: float | None, parsed: dict[str, float] | None, key: str, default: float | None) -> float | None:
    if explicit is not None:
        return explicit
    if parsed is not None:
        return parsed[key]
    return default


def _missing_split_fields(*, split_contract: str | Path | None, split_method: str | None, split_ratio: str | None) -> list[str]:
    if split_contract is not None:
        return []
    if split_method is None:
        return ["split_method"]
    normalized = str(split_method).strip().lower().replace("-", "_")
    split_type = _split_type_for_method(normalized)
    if split_type == "holdout" and normalized in {"auto", "random", "stratified"} and split_ratio is None:
        return ["split_ratio"]
    return []
