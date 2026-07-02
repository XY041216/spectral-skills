"""Spectral-reader runtime health for one-shot data reading."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
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
SKILL_DIR = Path(__file__).resolve().parents[1]

from spectral_core.reader.response import ok_response
from spectral_core.reader.version import CORE_VERSION, SCHEMA_VERSION

REQUIRED_TOOLS = ["server_health", "read_spectral_dataset", "check_consistency"]


def _optional_dependencies() -> tuple[dict[str, str], list[str]]:
    deps = {}
    missing = []
    packages = {
        "pandas": "pandas",
        "openpyxl": "openpyxl",
        "xlrd": "xlrd",
        "odfpy": "odf",
        "numpy": "numpy",
        "scipy": "scipy",
        "h5py": "h5py",
        "netCDF4": "netCDF4",
        "jsonschema": "jsonschema",
        "yaml": "yaml",
    }
    for display_name, module_name in packages.items():
        status = "ok" if importlib.util.find_spec(module_name) is not None else "missing"
        deps[display_name] = status
        if status == "missing":
            missing.append(display_name)
    return deps, missing


def _output_write_check() -> dict[str, str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "write_check.txt"
            path.write_text("ok", encoding="utf-8")
            return {"status": "ok"}
    except Exception as exc:
            return {"status": "failed", "message": str(exc)}


def _tool_availability() -> tuple[dict[str, bool], dict[str, bool]]:
    skill_scripts = {}
    fallback_scripts = {}
    for tool in REQUIRED_TOOLS:
        skill_scripts[tool] = (SKILL_DIR / "scripts" / f"{tool}.py").exists()
        fallback_scripts[tool] = (ROOT / "scripts" / "reader" / f"{tool}.py").exists()
    return skill_scripts, fallback_scripts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check spectral-reader one-shot reading health.")
    parser.add_argument("--json", action="store_true")
    parser.parse_args(argv)
    deps, missing = _optional_dependencies()
    skill_scripts, fallback_scripts = _tool_availability()
    schema_files = [
        "spectral_data_contract.schema.json",
        "read_spectral_dataset_result.schema.json",
    ]
    schema_dir = SKILL_DIR / "schemas"
    schemas_ok = all((schema_dir / name).exists() for name in schema_files)
    core_status = "healthy" if all(skill_scripts.values()) and schemas_ok else "degraded"
    response = ok_response(
        "server_health",
        {
            "status": core_status if not missing else "degraded",
            "stage": "reader_one_shot_runtime",
            "server_health_available": skill_scripts["server_health"],
            "read_spectral_dataset_available": skill_scripts["read_spectral_dataset"],
            "check_consistency_available": skill_scripts["check_consistency"],
            "reader_core_available": True,
            "schemas_available": schemas_ok,
            "data_contract_schema_available": (schema_dir / "spectral_data_contract.schema.json").exists(),
            "read_spectral_dataset_result_schema_available": (schema_dir / "read_spectral_dataset_result.schema.json").exists(),
            "required_schemas_available": schemas_ok,
            "mcp_tools_declared": ["reader.server_health", "reader.read_spectral_dataset"],
            "primary_entrypoint": "read_spectral_dataset",
            "one_shot_read_available": skill_scripts["read_spectral_dataset"],
            "supported_input_types": ["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "ods", "npy", "npz", "mat", "h5", "hdf5", "nc", "folder"],
            "openpyxl_available": deps.get("openpyxl") == "ok",
            "xlrd_available": deps.get("xlrd") == "ok",
            "odfpy_available": deps.get("odfpy") == "ok",
            "excel_reading_available": deps.get("pandas") == "ok" and deps.get("openpyxl") == "ok",
            "ods_reading_available": deps.get("pandas") == "ok" and deps.get("odfpy") == "ok",
            "numpy_available": deps.get("numpy") == "ok",
            "scipy_available": deps.get("scipy") == "ok",
            "npy_reading_available": deps.get("numpy") == "ok",
            "npz_reading_available": deps.get("numpy") == "ok",
            "mat_reading_available": deps.get("numpy") == "ok" and deps.get("scipy") == "ok",
            "h5py_available": deps.get("h5py") == "ok",
            "netcdf4_available": deps.get("netCDF4") == "ok",
            "hdf5_reading_available": deps.get("h5py") == "ok" and deps.get("numpy") == "ok",
            "netcdf_reading_available": deps.get("netCDF4") == "ok" and deps.get("numpy") == "ok",
            "output_write_check": _output_write_check(),
            "standard_output_files": ["X.csv", "y.csv", "sample_ids.csv", "band_axis.csv", "metadata.csv", "data_contract.json"],
            "forbidden_default_outputs": ["package_manifest.json", "summary.json", "_internal", "preview_report.json", "validation_report.json", "profile_summary.json", "variable_inventory.json", "dataset_inventory.json", "logs"],
            "standard_output_version": "standard_output.v1",
            "required_dependencies": {"python_stdlib": "ok"},
            "optional_dependencies": deps,
            "missing_optional_dependencies": missing,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "reader_core_version": CORE_VERSION,
            "schema_version": SCHEMA_VERSION,
        },
        backend="script",
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
