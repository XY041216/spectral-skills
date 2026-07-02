"""Writers for the minimal spectral-reader standard output."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from .assertions import (
    ReaderAssertionError,
    assert_X_exists,
    assert_X_numeric,
    assert_band_axis_aligned,
    assert_metadata_aligned,
    assert_non_empty_matrix,
    assert_required_refs_exist,
    assert_sample_ids_aligned,
    assert_y_aligned,
)
from .io_utils import load_json_file, write_json_file
from .response import error_response, ok_response


STANDARD_FILES = ["X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"]


def write_apply_outputs(
    *,
    output_dir: str | Path,
    X: list[list[float]],
    spectral_columns: list[str],
    y: list[Any] | None,
    y_name: str | list[str] | None,
    sample_ids: list[Any] | None,
    band_axis: list[Any],
    band_unit: str | None,
    metadata_rows: list[dict[str, Any]] | None,
    apply_result: dict[str, Any],
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write apply outputs for helper-tool use.

    The standalone apply tool keeps its data under data/ so older helper tools can
    load it, but it does not write logs, validation reports, manifests, or traces.
    The main read_spectral_dataset entry writes the final flat user output.
    """

    root = Path(output_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    refs: dict[str, str] = {}
    _write_rows(data_dir / "X.csv", [list(spectral_columns), *X])
    refs["X_ref"] = "data/X.csv"

    if y is not None:
        _write_rows(data_dir / "y.csv", _y_rows(y, y_name))
        refs["y_ref"] = "data/y.csv"

    if sample_ids is not None:
        _write_rows(data_dir / "sample_ids.csv", [["sample_id"], *[[value] for value in sample_ids]])
        refs["sample_ids_ref"] = "data/sample_ids.csv"

    _write_rows(data_dir / "band_axis.csv", [["index", "value", "unit"], *[[idx, value, band_unit] for idx, value in enumerate(band_axis)]])
    refs["band_axis_ref"] = "data/band_axis.csv"

    if metadata_rows is not None:
        _write_dict_rows(data_dir / "metadata.csv", metadata_rows)
        refs["metadata_ref"] = "data/metadata.csv"

    apply_result_path = root / "apply_read_plan_result.json"
    apply_result_with_refs = dict(apply_result)
    apply_result_with_refs["data_refs"] = dict(refs)
    write_json_file(apply_result_path, apply_result_with_refs, ensure_ascii=False)
    refs["apply_result_ref"] = "apply_read_plan_result.json"
    return refs


def write_contract_outputs(
    *,
    output: str | Path,
    data_contract: dict[str, Any],
    profile_summary: dict[str, Any] | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write only data_contract.json for compatibility with build_data_contract."""

    contract_path = Path(output)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(contract_path, data_contract, ensure_ascii=False)
    return {"data_contract_ref": str(contract_path)}


def write_standardized_package(
    *,
    apply_dir: str | Path | None = None,
    contract_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    read_plan: str | Path | None = None,
    preview_report: str | Path | None = None,
    make_zip: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    backend: str = "core",
) -> dict[str, Any]:
    """Write the final user-facing reader output directory.

    The output is flat: X.csv, optional y.csv, sample_ids.csv, band_axis.csv,
    metadata.csv when available, and data_contract.json. No manifest, summary,
    _internal tree, logs, reports, or persisted preview/read_plan artifacts are
    generated.
    """

    if apply_dir is None or contract_dir is None or output_dir is None:
        return error_response(
            "write_standardized_package",
            "apply_dir, contract_dir, and output_dir are required.",
            backend=backend,
            code="STANDARD_OUTPUT_INPUT_MISSING",
        )
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        return _blocked("output_dir already exists and is not empty. Use --overwrite to rebuild.", "OUTPUT_DIR_EXISTS", backend, output_root)
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        copied = _copy_standard_files(Path(apply_dir), Path(contract_dir), output_root)
        contract_path = output_root / "data_contract.json"
        contract = load_json_file(contract_path)
        _assert_standard_output(output_root, contract)
        result = _standard_result(output_root, contract, copied)
        return ok_response("write_standardized_package", result, backend=backend, warnings=result.get("warnings", []))
    except ReaderAssertionError as exc:
        return _blocked(exc.message, exc.code, backend, output_root, **exc.details)
    except Exception as exc:
        return _blocked(f"Could not write standardized reader output: {exc}", "STANDARD_OUTPUT_WRITE_FAILED", backend, output_root)


def validate_package_structure(package_dir: str | Path) -> dict[str, Any]:
    """Validate the minimal standard reader output directory."""

    root = Path(package_dir)
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not root.exists():
        return {"package_status": "blocked", "blocking_errors": [{"code": "OUTPUT_DIR_NOT_FOUND", "message": "output directory does not exist.", "details": {"path": str(root)}}], "warnings": []}
    contract_path = root / "data_contract.json"
    if not contract_path.exists():
        blocking.append({"code": "DATA_CONTRACT_MISSING", "message": "data_contract.json is missing.", "details": {}})
        contract = {}
    else:
        try:
            contract = load_json_file(contract_path)
        except Exception as exc:
            contract = {}
            blocking.append({"code": "DATA_CONTRACT_UNREADABLE", "message": "data_contract.json could not be parsed.", "details": {"error": str(exc)}})
    try:
        if contract:
            _assert_standard_output(root, contract)
    except ReaderAssertionError as exc:
        blocking.append(exc.as_issue())
    for forbidden in ["package_manifest.json", "summary.json", "_internal", "logs", "preview_report.json", "validation_report.json", "profile_summary.json"]:
        if (root / forbidden).exists():
            warnings.append({"code": "REDUNDANT_OUTPUT_PRESENT", "message": f"Redundant output should not be generated: {forbidden}", "details": {"path": forbidden}})
    return {
        "package_status": "blocked" if blocking else "ready",
        "blocking_errors": blocking,
        "warnings": warnings,
        "contract": contract,
    }


def write_package(contract: dict[str, Any] | None = None, output_dir: str | None = None, *, backend: str = "core") -> dict[str, Any]:
    return ok_response(
        "write_package",
        {
            "status": "compatibility_only",
            "output_dir": output_dir,
            "message": "spectral-reader writes standard output files directly; no package manifest is produced.",
        },
        backend=backend,
    )


def _copy_standard_files(apply_root: Path, contract_root: Path, output_root: Path) -> list[str]:
    refs = _load_apply_refs(apply_root)
    copied: list[str] = []
    for ref_key, output_name, required in [
        ("X_ref", "X.csv", True),
        ("y_ref", "y.csv", False),
        ("sample_ids_ref", "sample_ids.csv", False),
        ("band_axis_ref", "band_axis.csv", True),
        ("metadata_ref", "metadata.csv", False),
    ]:
        ref = refs.get(ref_key)
        src = _resolve_ref(apply_root, ref) if ref else None
        if src is None or not src.exists():
            if required:
                raise ReaderAssertionError("STANDARD_FILE_MISSING", f"{output_name} is missing from apply outputs.", ref_key=ref_key)
            continue
        shutil.copy2(src, output_root / output_name)
        copied.append(output_name)

    contract_src = contract_root / "data_contract.json"
    if not contract_src.exists():
        raise ReaderAssertionError("DATA_CONTRACT_MISSING", "data_contract.json is missing from contract outputs.")
    contract = load_json_file(contract_src)
    flat_contract = _rebase_contract(contract, output_root)
    write_json_file(output_root / "data_contract.json", flat_contract, ensure_ascii=False)
    copied.append("data_contract.json")
    return copied


def _load_apply_refs(apply_root: Path) -> dict[str, str]:
    apply_result_path = apply_root / "apply_read_plan_result.json"
    if not apply_result_path.exists():
        refs: dict[str, str] = {}
        for name, key in [("X.csv", "X_ref"), ("y.csv", "y_ref"), ("sample_ids.csv", "sample_ids_ref"), ("band_axis.csv", "band_axis_ref"), ("metadata.csv", "metadata_ref")]:
            if (apply_root / name).exists():
                refs[key] = name
            elif (apply_root / "data" / name).exists():
                refs[key] = f"data/{name}"
        return refs
    apply_result = load_json_file(apply_result_path)
    return dict(apply_result.get("data_refs") or {})


def _rebase_contract(contract: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rebased = dict(contract)
    existing = {path.name for path in output_root.iterdir() if path.is_file()}
    files = dict(rebased.get("files") or {})
    files["X"] = "X.csv"
    files["band_axis"] = "band_axis.csv"
    files["sample_ids"] = "sample_ids.csv" if "sample_ids.csv" in existing else None
    files["metadata"] = "metadata.csv" if "metadata.csv" in existing else None
    files["y"] = "y.csv" if "y.csv" in existing else None
    rebased["files"] = files
    rebased["X"] = "X.csv"
    rebased["band_axis_ref"] = "band_axis.csv"
    axis = rebased.get("band_axis")
    if isinstance(axis, dict):
        axis = dict(axis)
        axis["file"] = "band_axis.csv"
        rebased["band_axis"] = axis
    else:
        rebased["band_axis"] = {"file": "band_axis.csv", "unit": rebased.get("band_unit"), "type": rebased.get("band_axis_type") or "unknown", "count": rebased.get("n_features")}
    rebased["sample_ids"] = "sample_ids.csv" if "sample_ids.csv" in existing else None
    rebased["metadata"] = "metadata.csv" if "metadata.csv" in existing else None
    if "y.csv" in existing:
        rebased["y"] = "y.csv"
        rebased["label_status"] = rebased.get("label_status") or "present"
    else:
        rebased["y"] = None
        rebased["label_status"] = rebased.get("label_status") or "absent"
    rebased["status"] = "ready" if rebased.get("status") in {None, "confirmed"} else rebased.get("status")
    return rebased


def _assert_standard_output(root: Path, contract: dict[str, Any]) -> None:
    files = dict(contract.get("files") or {})
    band_axis_value = contract.get("band_axis")
    band_axis_ref = files.get("band_axis") or contract.get("band_axis_ref")
    if not band_axis_ref and isinstance(band_axis_value, dict):
        band_axis_ref = band_axis_value.get("file")
    elif not band_axis_ref:
        band_axis_ref = band_axis_value
    refs = {
        "X_ref": files.get("X") or contract.get("X"),
        "y_ref": files.get("y") or contract.get("y"),
        "sample_ids_ref": files.get("sample_ids") or contract.get("sample_ids"),
        "band_axis_ref": band_axis_ref,
        "metadata_ref": files.get("metadata") or contract.get("metadata"),
    }
    assert_required_refs_exist(root, refs, ["X_ref", "band_axis_ref"])
    x_path = assert_X_exists(root, refs.get("X_ref"))
    assert_X_numeric(x_path)
    n_samples, n_features = assert_non_empty_matrix(x_path)
    assert_band_axis_aligned(root, refs.get("band_axis_ref"), n_features)
    assert_y_aligned(root, refs.get("y_ref"), n_samples)
    assert_sample_ids_aligned(root, refs.get("sample_ids_ref"), n_samples)
    assert_metadata_aligned(root, refs.get("metadata_ref"), n_samples)


def _standard_result(output_root: Path, contract: dict[str, Any], copied: list[str]) -> dict[str, Any]:
    files = dict(contract.get("files") or {})
    shape = dict(contract.get("shape") or {})
    band_axis = contract.get("band_axis") if isinstance(contract.get("band_axis"), dict) else {}
    return {
        "status": contract.get("status") or "ready",
        "output_dir": str(output_root),
        "X": files.get("X") or contract.get("X"),
        "y": files.get("y") or contract.get("y"),
        "sample_ids": files.get("sample_ids") or contract.get("sample_ids"),
        "band_axis": files.get("band_axis") or contract.get("band_axis_ref") or band_axis.get("file"),
        "metadata": files.get("metadata") or contract.get("metadata"),
        "data_contract": "data_contract.json",
        "n_samples": shape.get("n_samples") or contract.get("n_samples"),
        "n_features": shape.get("n_features") or contract.get("n_features"),
        "label_status": contract.get("label_status"),
        "task_hint": contract.get("task_hint"),
        "band_unit": band_axis.get("unit") or contract.get("band_unit"),
        "spectral_type": contract.get("spectral_type"),
        "source_type": contract.get("source_type"),
        "written_files": copied,
        "warnings": [],
        "next_step_hint": "Use data_contract.json and the standard CSV files as downstream input.",
    }


def _resolve_ref(root: Path, ref: str | None) -> Path | None:
    if not ref:
        return None
    path = Path(ref)
    return path if path.is_absolute() else root / path


def _write_rows(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _y_rows(y: list[Any], y_name: str | list[str] | None) -> list[list[Any]]:
    if isinstance(y_name, list):
        header = [str(value) for value in y_name]
        rows = [list(value) if isinstance(value, (list, tuple)) else [value] for value in y]
        return [header, *rows]
    header = [str(y_name or "y")]
    if y and isinstance(y[0], (list, tuple)):
        width = len(y[0])
        header = [str(y_name or f"target_{idx + 1}") if width == 1 else f"target_{idx + 1}" for idx in range(width)]
        return [header, *[list(value) for value in y]]
    return [header, *[[value] for value in y]]


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _blocked(message: str, code: str, backend: str, output_root: Path, **details: Any) -> dict[str, Any]:
    return error_response(
        "write_standardized_package",
        message,
        backend=backend,
        code=code,
        result={"status": "blocked", "output_dir": str(output_root), "blocking_reason": message},
        details=details,
    )
