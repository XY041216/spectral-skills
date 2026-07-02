from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from spectral_core.workflow.workflow import run_spectral_workflow


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_raw_classification(path: Path) -> Path:
    _write_rows(
        path,
        [
            ["sample_id", "900", "1000", "1100", "class"],
            ["S001", 1, 2, 3, "A"],
            ["S002", 2, 3, 4, "A"],
            ["S003", 3, 4, 5, "A"],
            ["S004", 4, 5, 6, "A"],
            ["S005", 8, 9, 10, "B"],
            ["S006", 9, 10, 11, "B"],
            ["S007", 10, 11, 12, "B"],
            ["S008", 11, 12, 13, "B"],
        ],
    )
    return path


def _write_package(root: Path) -> Path:
    _write_rows(
        root / "X.csv",
        [
            ["900", "1000", "1100"],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [8, 9, 10],
            [9, 10, 11],
            [10, 11, 12],
            [11, 12, 13],
        ],
    )
    _write_rows(root / "sample_ids.csv", [["sample_id"], ["S001"], ["S002"], ["S003"], ["S004"], ["S005"], ["S006"], ["S007"], ["S008"]])
    _write_rows(root / "band_axis.csv", [["index", "value", "unit"], [0, 900, "nm"], [1, 1000, "nm"], [2, 1100, "nm"]])
    _write_rows(root / "y.csv", [["class"], ["A"], ["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["B"]])
    contract = {
        "contract_id": "workflow-package-test",
        "files": {"X": "X.csv", "sample_ids": "sample_ids.csv", "band_axis": "band_axis.csv", "y": "y.csv", "metadata": None},
        "shape": {"n_samples": 8, "n_features": 3},
        "task_hint": "classification",
    }
    (root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return root


def _fake_stage_runner(output_dir: Path, calls: list[tuple[str, list[str]]] | None = None):
    def fake_stage_runner(stage: str, args: list[str]) -> dict[str, object]:
        if calls is not None:
            calls.append((stage, args))
        if stage == "reader":
            _write_package(Path(args[args.index("--output-dir") + 1]))
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage == "qc":
            qc_dir = Path(args[args.index("--output-dir") + 1])
            qc_dir.mkdir(parents=True, exist_ok=True)
            (qc_dir / "qc_result.json").write_text(json.dumps({"stage": "spectral-qc", "status": "passed", "checks": []}), encoding="utf-8")
            return {"ok": True, "result": {"status": "passed", "checks": []}, "warnings": [], "errors": []}
        if stage == "splitter":
            split_dir = Path(args[args.index("--output-dir") + 1])
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "split_contract.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage in {"preprocess", "feature"}:
            package_dir = Path(args[args.index("--output-dir") + 1])
            _write_package(package_dir)
            if stage == "feature":
                (package_dir / "feature_state.json").write_text(
                    json.dumps({"state_type": "feature_state", "methods": ["none"], "input_n_features": 3, "output_n_features": 3}),
                    encoding="utf-8",
                )
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage == "modeling":
            model_dir = Path(args[args.index("--output-dir") + 1])
            model_dir.mkdir(parents=True, exist_ok=True)
            if "--mode" in args and args[args.index("--mode") + 1] == "repeated_classifier_comparison":
                (model_dir / "classifier_comparison_contract.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            else:
                (model_dir / "modeling_contract.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        raise AssertionError(stage)

    return fake_stage_runner


def _fake_stage_runner_with_stale_feature_contract(output_dir: Path):
    def fake_stage_runner(stage: str, args: list[str]) -> dict[str, object]:
        if stage == "splitter":
            split_dir = Path(args[args.index("--output-dir") + 1])
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "split_contract.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage == "preprocess":
            package_dir = Path(args[args.index("--output-dir") + 1])
            _write_package(package_dir)
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage == "feature":
            package_dir = Path(args[args.index("--output-dir") + 1])
            _write_package(package_dir)
            contract_path = package_dir / "data_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["n_features"] = 5
            contract["band_axis"] = {"file": "band_axis.csv", "count": 5, "type": "derived_feature_axis"}
            contract["spectral"] = {"n_bands": 5}
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            (package_dir / "feature_state.json").write_text(
                json.dumps({"state_type": "feature_state", "methods": ["pca"], "input_n_features": 5, "output_n_features": 3}),
                encoding="utf-8",
            )
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        if stage == "modeling":
            model_dir = Path(args[args.index("--output-dir") + 1])
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "modeling_contract.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            return {"ok": True, "result": {"status": "ready"}, "warnings": [], "errors": []}
        raise AssertionError(stage)

    return fake_stage_runner


def _fake_stage_runner_with_cleaned_qc(output_dir: Path, calls: list[tuple[str, list[str]]]):
    def fake_stage_runner(stage: str, args: list[str]) -> dict[str, object]:
        calls.append((stage, args))
        if stage == "qc":
            qc_dir = Path(args[args.index("--output-dir") + 1])
            cleaned = qc_dir / "cleaned_package"
            _write_package(cleaned)
            qc_dir.mkdir(parents=True, exist_ok=True)
            (qc_dir / "qc_result.json").write_text(
                json.dumps(
                    {
                        "stage": "spectral-qc",
                        "status": "cleaned",
                        "output_package": str(cleaned),
                        "next_package_for_downstream": str(cleaned),
                    }
                ),
                encoding="utf-8",
            )
            return {"ok": True, "result": {"status": "cleaned", "next_package_for_downstream": str(cleaned)}, "warnings": [], "errors": []}
        return _fake_stage_runner(output_dir)(stage, args)

    return fake_stage_runner


def test_raw_reader_splitter_modeling_workflow(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="none",
        models="random_forest_classifier",
        overwrite=True,
        stage_runner=_fake_stage_runner(output_dir),
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["workflow_status"] == "ready"
    assert set(result["stage_outputs"]) == {"reader", "qc", "splitter", "preprocess", "feature", "modeling"}
    assert result["stage_outputs"]["preprocess"] == "skipped_none"
    assert result["stage_outputs"]["feature"] == "skipped_none"
    assert result["final_output"].endswith("modeling_contract.json")
    assert (output_dir / "workflow_result.json").exists()
    assert (output_dir / "workflow_plan.json").exists()


def test_workflow_forwards_reader_wide_table_arguments(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="none",
        models="random_forest_classifier",
        reader_sample_orientation="rows",
        reader_sample_id_column_index=0,
        reader_label_column="class",
        reader_spectral_start_column="900",
        reader_spectral_end_column="1100",
        reader_band_type="wavelength",
        reader_band_unit="nm",
        reader_max_auto_columns=12000,
        reader_max_spectral_columns=24000,
        reader_wide_table_mode="auto",
        reader_confirm_read_plan=True,
        overwrite=True,
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    reader_args = next(args for stage, args in calls if stage == "reader")
    assert reader_args[reader_args.index("--sample-orientation") + 1] == "rows"
    assert reader_args[reader_args.index("--sample-id-column-index") + 1] == "0"
    assert reader_args[reader_args.index("--label-column") + 1] == "class"
    assert reader_args[reader_args.index("--spectral-start-column") + 1] == "900"
    assert reader_args[reader_args.index("--spectral-end-column") + 1] == "1100"
    assert reader_args[reader_args.index("--band-type") + 1] == "wavelength"
    assert reader_args[reader_args.index("--band-unit") + 1] == "nm"
    assert reader_args[reader_args.index("--max-auto-columns") + 1] == "12000"
    assert reader_args[reader_args.index("--max-spectral-columns") + 1] == "24000"
    assert reader_args[reader_args.index("--wide-table-mode") + 1] == "auto"
    assert "--confirm-read-plan" in reader_args


def test_run_workflow_cli_accepts_reader_wide_table_aliases(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow_cli"
    script = REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "run_spectral_workflow.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(raw),
            "--output-dir",
            str(output_dir),
            "--task-goal",
            "read",
            "--reader-sample-orientation",
            "rows",
            "--reader-sample-id-column-index",
            "0",
            "--reader-label-column",
            "class",
            "--reader-spectral-start-column",
            "900",
            "--reader-spectral-end-column",
            "1100",
            "--reader-band-type",
            "wavelength",
            "--reader-band-unit",
            "nm",
            "--max-auto-columns",
            "12000",
            "--wide-table-mode",
            "auto",
            "--reader-confirm-read-plan",
            "--overwrite",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["reader_max_auto_columns"] == 12000
    assert manifest["parameters"]["reader_wide_table_mode"] == "auto"
    assert (output_dir / "reader_package" / "data_contract.json").exists()


def test_package_qc_splitter_modeling_workflow(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        include_qc=True,
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="none",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir),
    )

    assert response["ok"] is True
    result = response["result"]
    assert "qc" in result["stage_outputs"]
    assert result["stage_outputs"]["qc"].endswith("qc_result.json")
    assert "splitter" in result["stage_outputs"]
    assert "modeling" in result["stage_outputs"]


def test_workflow_uses_cleaned_qc_package_for_downstream_splitter(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="split",
        include_qc=True,
        qc_mode="clean",
        split_ratio="6:2:2",
        split_method="random",
        stage_runner=_fake_stage_runner_with_cleaned_qc(output_dir, calls),
    )

    assert response["ok"] is True
    splitter_args = next(args for stage, args in calls if stage == "splitter")
    assert splitter_args[splitter_args.index("--package-dir") + 1] == str(output_dir / "qc_output" / "cleaned_package")


def test_prepare_for_optimizer_stops_after_reader_qc_split(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="prepare_for_optimizer",
        split_ratio="6:2:2",
        split_method="stratified",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    assert [stage for stage, _ in calls] == ["reader", "qc", "splitter"]
    assert set(response["result"]["stage_outputs"]) == {"reader", "qc", "splitter"}
    assert response["result"]["final_output"].endswith("split_contract.json")


def test_full_preprocess_feature_modeling_workflow(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="none",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir),
    )

    assert response["ok"] is True
    result = response["result"]
    assert set(result["stage_outputs"]) == {"reader", "qc", "splitter", "preprocess", "feature", "modeling"}
    assert result["stage_outputs"]["reader"].startswith("reused_from:")
    assert result["stage_outputs"]["preprocess"] == "skipped_none"
    assert result["stage_outputs"]["feature"] == "skipped_none"
    contract = json.loads((output_dir / "workflow_result.json").read_text(encoding="utf-8"))
    assert contract["final_output"].endswith("modeling_contract.json")


def test_workflow_repairs_feature_contract_before_modeling(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        skip_qc=True,
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="pca",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner_with_stale_feature_contract(output_dir),
    )

    assert response["ok"] is True
    assert response["warnings"][0]["code"] == "FEATURE_CONTRACT_REPAIRED"
    feature_contract = json.loads((output_dir / "feature_output" / "data_contract.json").read_text(encoding="utf-8"))
    assert feature_contract["n_features"] == 3
    assert feature_contract["band_axis"]["count"] == 3
    assert feature_contract["spectral"]["n_bands"] == 3
    assert feature_contract["source_spectral"]["original_n_bands"] == 5


def test_workflow_orchestrates_child_skill_entries(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="classification",
        include_qc=True,
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="snv",
        feature_method="none",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    assert [stage for stage, _ in calls] == ["reader", "qc", "splitter", "preprocess", "modeling"]
    assert calls[0][1][0:4] == ["--input", str(raw), "--output-dir", str(output_dir / "reader_package")]
    assert "--split-contract" in calls[3][1]
    assert calls[4][1][calls[4][1].index("--package-dir") + 1] == str(output_dir / "preprocess_output")


def test_workflow_passes_preprocess_and_feature_parameters(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="snv,band_range_select",
        preprocess_band_range="900:1000",
        confirm_band_change=True,
        feature_method="pca",
        feature_n_components=2,
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    preprocess_args = next(args for stage, args in calls if stage == "preprocess")
    feature_args = next(args for stage, args in calls if stage == "feature")
    assert preprocess_args[preprocess_args.index("--methods") + 1] == "snv,band_range_select"
    assert preprocess_args[preprocess_args.index("--band-range") + 1] == "900:1000"
    assert "--confirm-band-change" in preprocess_args
    assert feature_args[feature_args.index("--method") + 1] == "pca"
    assert feature_args[feature_args.index("--n-components") + 1] == "2"
    plan = json.loads((output_dir / "workflow_plan.json").read_text(encoding="utf-8"))
    preprocess_stage = next(stage for stage in plan["stages"] if stage["stage"] == "preprocess")
    feature_stage = next(stage for stage in plan["stages"] if stage["stage"] == "feature")
    assert preprocess_stage["parameters"]["band_range"] == "900:1000"
    assert feature_stage["parameters"]["n_components"] == 2


def test_workflow_output_root_creates_managed_run_layout(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "Tablet_ext_0-3.csv")
    output_root = tmp_path / "spectral_runs"
    run_dir = output_root / "Tablet_ext_0-3" / "manual_kfold5_snv_pca10_svm"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_root=output_root,
        run_name="manual_kfold5_snv_pca10_svm",
        task_goal="classification",
        split_method="stratified_kfold",
        n_splits=5,
        preprocess_methods="snv",
        feature_method="pca",
        feature_n_components=10,
        models="svm",
        stage_runner=_fake_stage_runner(run_dir, calls),
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["run_id"] == "manual_kfold5_snv_pca10_svm"
    assert result["dataset_name"] == "Tablet_ext_0-3"
    assert Path(result["run_dir"]) == run_dir.resolve()
    assert Path(result["output_root"]) == output_root.resolve()
    assert (run_dir / "workflow_plan.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "reader_package" / "data_contract.json").exists()
    assert (run_dir / "qc_output" / "qc_result.json").exists()
    assert (run_dir / "split_output" / "split_contract.json").exists()
    assert (run_dir / "preprocess_output" / "data_contract.json").exists()
    assert (run_dir / "feature_output" / "data_contract.json").exists()
    assert (run_dir / "model_output" / "modeling_contract.json").exists()
    assert result["stage_outputs_relative"]["reader"] == "reader_package/data_contract.json"
    assert result["stage_outputs_relative"]["qc"] == "qc_output/qc_result.json"
    assert result["stage_outputs_relative"]["splitter"] == "split_output/split_contract.json"
    assert (output_root / "Tablet_ext_0-3" / "runs_index.csv").exists()
    assert (output_root / "Tablet_ext_0-3" / "latest.txt").read_text(encoding="utf-8") == "manual_kfold5_snv_pca10_svm"


def test_managed_layout_uses_source_input_name_for_reused_reader_package(tmp_path: Path) -> None:
    previous = tmp_path / "previous_run"
    package_dir = _write_package(previous / "reader_package")
    contract_path = package_dir / "data_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source"] = {"input": str(tmp_path / "Tablet_ext_0-3.csv")}
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    output_root = tmp_path / "spectral_runs"
    run_name = "reused_package_split"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_root=output_root,
        run_name=run_name,
        task_goal="split",
        split_method="stratified_monte_carlo_cv",
        n_repeats=10,
        train_ratio=0.7,
        test_ratio=0.3,
        skip_qc=True,
        stage_runner=_fake_stage_runner(output_root),
    )

    assert response["ok"] is True
    assert response["result"]["dataset_name"] == "Tablet_ext_0-3"
    assert Path(response["result"]["run_dir"]) == (output_root / "Tablet_ext_0-3" / run_name).resolve()


def test_workflow_routes_repeated_multiclassifier_run_to_classifier_comparison_mode(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        skip_qc=True,
        split_method="stratified_monte_carlo_cv",
        n_repeats=10,
        train_ratio=0.7,
        test_ratio=0.3,
        preprocess_methods="snv",
        feature_method="pca",
        feature_explained_variance=0.95,
        auto_confirm_feature_defaults=True,
        models="regular-fast",
        auto_confirm_model_defaults=True,
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    modeling_args = next(args for stage, args in calls if stage == "modeling")
    assert modeling_args[modeling_args.index("--mode") + 1] == "repeated_classifier_comparison"
    assert "--disable-model-selection" in modeling_args
    assert "--checkpoint-per-model" in modeling_args
    assert modeling_args[modeling_args.index("--candidate-model-set-source") + 1] == "workflow_auto_repeated_classifier_comparison"
    assert response["result"]["final_output"].endswith("classifier_comparison_contract.json")


def test_workflow_skip_qc_opt_out(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="classification",
        skip_qc=True,
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="none",
        feature_method="none",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    assert "qc" not in response["result"]["stage_outputs"]
    assert [stage for stage, _ in calls] == ["reader", "splitter", "modeling"]


def test_workflow_infers_feature_skip_for_preprocess_to_model_pipeline(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:2",
        split_method="random",
        preprocess_methods="snv",
        models="random_forest_classifier",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is True
    assert [stage for stage, _ in calls] == ["qc", "splitter", "preprocess", "modeling"]
    result = response["result"]
    assert result["stage_outputs"]["feature"] == "skipped_none"
    plan = json.loads((output_dir / "workflow_plan.json").read_text(encoding="utf-8"))
    feature_stage = next(stage for stage in plan["stages"] if stage["stage"] == "feature")
    assert feature_stage["status"] == "skip"
    assert feature_stage["decision_source"] == "inferred_from_user_pipeline"


def test_cv_preprocess_contract_flows_directly_to_modeling_when_feature_skipped(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_method="stratified_kfold",
        n_splits=2,
        preprocess_methods="snv",
        feature_method="none",
        models="random_forest_classifier",
        overwrite=True,
        backend="core",
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["stage_outputs"]["preprocess"].endswith("preprocess_contract.json")
    assert result["stage_outputs"]["feature"] == "skipped_none"
    assert result["stage_outputs"]["modeling"].endswith("cv_modeling_result.json")
    assert (output_dir / "model_output" / "fold_metrics.csv").exists()


def test_missing_goal_and_split_ratio_need_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")

    missing_goal = run_spectral_workflow(package_dir=package_dir, output_dir=tmp_path / "wg")
    assert missing_goal["ok"] is False
    assert missing_goal["result"]["status"] == "needs_confirmation"
    assert missing_goal["errors"][0]["code"] == "WORKFLOW_GOAL_REQUIRED"

    missing_method = run_spectral_workflow(package_dir=package_dir, output_dir=tmp_path / "wm", task_goal="classification", models="random_forest_classifier")
    assert missing_method["ok"] is False
    assert missing_method["result"]["status"] == "needs_confirmation"
    assert missing_method["errors"][0]["code"] == "SPLIT_METHOD_REQUIRED"

    missing_ratio = run_spectral_workflow(package_dir=package_dir, output_dir=tmp_path / "wr", task_goal="classification", split_method="random", models="random_forest_classifier")
    assert missing_ratio["ok"] is False
    assert missing_ratio["result"]["status"] == "needs_confirmation"
    assert missing_ratio["errors"][0]["code"] == "SPLIT_RATIO_REQUIRED"


def test_incomplete_split_ratio_blocks_before_reader_or_splitter(tmp_path: Path) -> None:
    raw = _write_raw_classification(tmp_path / "raw.csv")
    output_dir = tmp_path / "workflow"
    calls: list[tuple[str, list[str]]] = []

    response = run_spectral_workflow(
        input_path=raw,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="6:2:",
        split_method="stratified",
        preprocess_methods="none",
        feature_method="none",
        models="svm",
        stage_runner=_fake_stage_runner(output_dir, calls),
    )

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "SPLIT_RATIO_CONFIRMATION_REQUIRED"
    assert response["result"]["confirmation_required"][0]["recommended_interpretation"] == "train:val:test = 6:2:2"
    assert calls == []
    assert not output_dir.exists()


def test_preprocess_goal_missing_method_needs_confirmation(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    response = run_spectral_workflow(package_dir=package_dir, output_dir=tmp_path / "workflow", task_goal="preprocess", split_ratio="8:2", split_method="random")

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "PREPROCESS_METHOD_REQUIRED"


def test_workflow_feature_key_parameters_need_confirmation_before_stages(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="8:2",
        split_method="stratified",
        preprocess_methods="none",
        feature_method="vip",
        models="svm",
    )

    assert response["ok"] is False
    assert response["result"]["status"] == "needs_confirmation"
    assert response["errors"][0]["code"] == "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED"
    assert {item["field"] for item in response["result"]["confirmation_required"]} == {"selection_rule", "n_components"}
    assert not (output_dir / "reader_package" / "data_contract.json").exists()
    assert not (output_dir / "split_output" / "split_contract.json").exists()


def test_workflow_uve_reports_all_key_parameters_at_once(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="classification",
        split_ratio="8:2",
        split_method="stratified",
        preprocess_methods="none",
        feature_method="uve",
        models="svm",
    )

    assert response["ok"] is False
    assert response["errors"][0]["code"] == "FEATURE_PARAMETERS_CONFIRMATION_REQUIRED"
    assert {item["field"] for item in response["result"]["confirmation_required"]} == {
        "n_components",
        "n_runs",
        "selection_rule",
        "random_state",
    }
    by_field = {item["field"]: item for item in response["result"]["confirmation_required"]}
    assert by_field["n_components"]["recommended"] == 10
    assert by_field["n_runs"]["recommended"] == 50
    assert by_field["random_state"]["recommended"] == 42
    assert "top_k=50" in by_field["selection_rule"]["options"]
    assert not (output_dir / "split_output" / "split_contract.json").exists()


def test_workflow_uve_partial_confirmation_reports_only_n_components(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=tmp_path / "workflow",
        task_goal="classification",
        split_ratio="8:2",
        split_method="stratified",
        preprocess_methods="none",
        feature_method="uve",
        feature_n_runs=50,
        feature_top_k=50,
        feature_random_state=42,
        models="svm",
    )

    assert response["ok"] is False
    assert [item["field"] for item in response["result"]["confirmation_required"]] == ["n_components"]


def test_workflow_cli_and_fallback_cli_emit_json(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    for idx, script in enumerate(
        [
            REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "run_spectral_workflow.py",
            REPO_ROOT / "scripts" / "workflow" / "run_spectral_workflow.py",
        ]
    ):
        output_dir = tmp_path / f"workflow_{idx}"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(output_dir),
                "--task-goal",
                "split",
                "--split-ratio",
                "6:2:2",
                "--split-method",
                "random",
                "--json",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert payload["result"]["workflow_result"].endswith("workflow_result.json")
        assert (output_dir / "workflow_result.json").exists()


def test_workflow_cli_output_root_runs_managed_pipeline(tmp_path: Path) -> None:
    package_dir = _write_package(tmp_path / "package")
    output_root = tmp_path / "spectral_runs"
    run_name = "cli_holdout_svm"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "run_spectral_workflow.py"),
            "--package-dir",
            str(package_dir),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--task-goal",
            "split",
            "--split-ratio",
            "6:2:2",
            "--split-method",
            "random",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    result = payload["result"]
    run_dir = output_root / "package" / run_name
    assert Path(result["run_dir"]) == run_dir.resolve()
    assert result["stage_outputs_relative"]["qc"] == "qc_output/qc_result.json"
    assert result["stage_outputs_relative"]["splitter"] == "split_output/split_contract.json"
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "logs" / "qc.log").exists()
    assert (output_root / "package" / "runs_index.csv").exists()
    assert (output_root / "package" / "latest.txt").read_text(encoding="utf-8") == run_name


def test_workflow_state_cli_tools_create_decision_and_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow_state"
    create = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "create_workflow_plan.py"),
            "--output-dir",
            str(output_dir),
            "--task-goal",
            "classification",
            "--package-dir",
            "package",
            "--include-qc",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    plan_path = output_dir / "workflow_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["workflow_plan_version"] == "0.2.0"
    assert plan["contract_chain"][1] == "qc_result.json"

    decision_file = output_dir / "split_decision.json"
    decision_file.write_text(json.dumps({"split_ratio": "6:2:2", "split_method": "stratified"}), encoding="utf-8")
    decision = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "workflow" / "update_workflow_decision.py"),
            "--plan",
            str(plan_path),
            "--stage",
            "splitter",
            "--decision-file",
            str(decision_file),
            "--decision",
            "random_seed=42",
            "--question",
            "Confirm split strategy including random_seed=42.",
            "--user-selected-option",
            "6:2:2 stratified seed=42",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert decision.returncode == 0, decision.stderr
    updated = json.loads(plan_path.read_text(encoding="utf-8"))
    assert updated["confirmation_log"][0]["stage"] == "splitter"
    splitter = next(item for item in updated["stages"] if item["stage"] == "splitter")
    assert splitter["parameters"]["method"] == "stratified"
    assert splitter["parameters"]["ratio"] == "6:2:2"
    assert splitter["confirmation"]["parameter_decisions"]["random_seed"]["decision_source"] == "recommended_default_confirmed_with_split"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "workflow" / "update_workflow_result.py"),
            "--result",
            str(output_dir / "workflow_result.json"),
            "--task-goal",
            "classification",
            "--stage-output-key",
            "qc",
            "--stage-output-path",
            "qc_output/qc_result.json",
            "--final-output",
            "model_output/modeling_contract.json",
            "--workflow-plan",
            str(plan_path),
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    workflow_result = json.loads((output_dir / "workflow_result.json").read_text(encoding="utf-8"))
    assert workflow_result["stage_outputs"]["qc"].endswith("qc_result.json")


def test_workflow_decision_updates_are_locked_and_atomic(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow_state_concurrent"
    create = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "create_workflow_plan.py"),
            "--output-dir",
            str(output_dir),
            "--task-goal",
            "classification",
            "--package-dir",
            "package",
            "--include-qc",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    plan_path = output_dir / "workflow_plan.json"
    script = REPO_ROOT / "scripts" / "workflow" / "update_workflow_decision.py"

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--plan",
                str(plan_path),
                "--stage",
                "splitter",
                "--decision-source",
                "concurrent_test",
                "--decision",
                f"random_seed={seed}",
                "--user-selected-option",
                f"seed={seed}",
                "--json",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for seed in range(40, 46)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr

    updated = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(updated["confirmation_log"]) == 6
    assert {item["parameters"]["random_seed"] for item in updated["confirmation_log"]} == set(range(40, 46))
    assert not plan_path.with_name("workflow_plan.json.lock").exists()


def test_workflow_plan_skips_none_and_uses_cv_parameters(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow_state_cv"
    create = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "create_workflow_plan.py"),
            "--output-dir",
            str(output_dir),
            "--task-goal",
            "classification",
            "--package-dir",
            "package",
            "--split-method",
            "stratified_kfold",
            "--preprocess-methods",
            "none",
            "--feature-method",
            "none",
            "--models",
            "svm",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    plan = json.loads((output_dir / "workflow_plan.json").read_text(encoding="utf-8"))
    splitter = next(item for item in plan["stages"] if item["stage"] == "splitter")
    preprocess = next(item for item in plan["stages"] if item["stage"] == "preprocess")
    feature = next(item for item in plan["stages"] if item["stage"] == "feature")
    assert splitter["status"] == "execute"
    assert splitter["parameters"]["split_type"] == "cross_validation"
    assert splitter["parameters"]["n_splits"] == 5
    assert "ratio" not in splitter["confirmation"]["required_fields"]
    assert preprocess["status"] == "skip"
    assert preprocess["decision_source"] == "user_specified"
    assert feature["status"] == "skip"
    assert feature["decision_source"] == "user_specified"


def test_workflow_plan_repeated_split_does_not_require_holdout_ratio(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow_state_mccv"
    create = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "skills" / "spectral-workflow" / "scripts" / "create_workflow_plan.py"),
            "--output-dir",
            str(output_dir),
            "--task-goal",
            "classification",
            "--package-dir",
            "package",
            "--split-method",
            "stratified_monte_carlo_cv",
            "--preprocess-methods",
            "none",
            "--feature-method",
            "none",
            "--models",
            "svm",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert create.returncode == 0, create.stderr
    plan = json.loads((output_dir / "workflow_plan.json").read_text(encoding="utf-8"))
    splitter = next(item for item in plan["stages"] if item["stage"] == "splitter")
    assert splitter["status"] == "execute"
    assert splitter["parameters"]["split_type"] == "repeated_holdout"
    assert splitter["parameters"]["n_repeats"] == 100
    assert splitter["parameters"]["train_ratio"] == 0.7
    assert splitter["parameters"]["test_ratio"] == 0.3
    assert splitter["confirmation"]["required_fields"] == []


def test_reused_package_records_reader_and_qc_provenance(tmp_path: Path) -> None:
    previous = tmp_path / "previous_workflow"
    package_dir = _write_package(previous / "reader_package")
    qc_dir = previous / "qc_output"
    qc_dir.mkdir(parents=True)
    (qc_dir / "qc_result.json").write_text(json.dumps({"status": "warning"}), encoding="utf-8")
    output_dir = tmp_path / "workflow"

    response = run_spectral_workflow(
        package_dir=package_dir,
        output_dir=output_dir,
        task_goal="split",
        split_ratio="6:2:2",
        split_method="stratified",
        stage_runner=_fake_stage_runner(output_dir),
    )

    assert response["ok"] is True
    outputs = response["result"]["stage_outputs"]
    assert outputs["reader"].startswith("reused_from:")
    assert outputs["qc"].startswith("reused_from:")
