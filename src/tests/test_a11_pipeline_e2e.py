import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, dataset_csv: Path, codes_json: Path) -> Path:
    config = {
        "seed": 42,
        "paths": {
            "dataset_csv": str(dataset_csv),
            "extracted_codes_json": str(codes_json),
            "outputs_metrics": "outputs/metrics",
            "outputs_models": "outputs/models",
            "outputs_viz": "outputs/visualizations",
            "outputs_preds": "outputs/predictions",
        },
        "data": {
            "image_size": [128, 128],
            "num_bands": 9,
            "normalization_method": "zscore",
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "model": {
            "base_config_name": "tl_baseline",
            "backbone": "mobilenetv2",
            "num_classes": 2,
            "resize_to": [160, 160],
            "dropout_rate": 0.25,
            "fine_tune_last_layers": 20,
        },
        "training": {
            "batch_size": 8,
            "head_epochs": 1,
            "head_learning_rate": 1.0e-4,
            "fine_tune_epochs": 1,
            "fine_tune_learning_rate": 1.0e-5,
            "early_stopping_patience_head": 1,
            "early_stopping_patience_ft": 1,
        },
        "evaluation": {
            "threshold_default": 0.5,
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
            ],
        },
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return path


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    pd = pytest.importorskip("pandas")

    dataset = pd.DataFrame({"image_id": ["A", "A", "B", "B"]})
    dataset_csv = tmp_path / "pixels_dataset.csv"
    dataset.to_csv(dataset_csv, index=False)

    codes_json = tmp_path / "codes.json"
    codes_json.write_text(
        json.dumps({"positivos": ["A"], "negativos": ["B"]}),
        encoding="utf-8",
    )
    return dataset_csv, codes_json


def _write_split_dataset(tmp_path: Path) -> tuple[Path, Path]:
    pd = pytest.importorskip("pandas")

    rows: list[dict[str, object]] = []
    positives = [f"POS_{index}" for index in range(5)]
    negatives = [f"NEG_{index}" for index in range(5)]

    for image_id in positives + negatives:
        for repeat in range(2):
            rows.append(
                {
                    "path": f"/dataset/{image_id}/tile_{repeat}.tif",
                    "pixel_0": float(repeat),
                    "count": 1,
                    "height": 1,
                    "width": 1,
                }
            )

    dataset = pd.DataFrame(rows)
    dataset_csv = tmp_path / "pixels_dataset.csv"
    dataset.to_csv(dataset_csv, index=False)

    codes_json = tmp_path / "codes.json"
    codes_json.write_text(
        json.dumps({"positivos": positives, "negativos": negatives}),
        encoding="utf-8",
    )
    return dataset_csv, codes_json


def _expected_summary_keys() -> set[str]:
    return {
        "config_used",
        "timestamp",
        "seed",
        "model_name",
        "evaluation_split",
        "threshold",
        "n_total",
        "n_valid",
        "n_train",
        "n_val",
        "n_test",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
        "test_roc_auc",
        "test_pr_auc",
        "training_time_seconds",
        "total_epochs",
        "head_epochs",
        "ft_epochs",
        "model_path",
        "history_path",
        "predictions_path",
        "experiment_dir",
    }


def _build_runtime_deps(split_data: dict[str, object]):
    pd = pytest.importorskip("pandas")

    from artefatos.a11_pipeline_e2e.src.evaluation import (
        build_summary,
        save_summary_files,
    )
    from artefatos.a11_pipeline_e2e.src.preprocessing import (
        ensure_output_dirs,
        load_pipeline_config,
        validate_input_files,
    )

    class DummyModel:
        pass

    def fake_prepare_split_data(config, limit_samples=None):
        assert limit_samples in {None, 2}
        return split_data

    def fake_run_training_pipeline(config, split_data, output_paths, skip_train=False):
        model_path = output_paths["models"] / "best_model.keras"
        model_path.write_text("dummy-model", encoding="utf-8")
        history_path = output_paths["models"] / "history.json"
        history_path.write_text(
            json.dumps({"loss": [0.5], "accuracy": [0.8], "val_accuracy": [0.75]}),
            encoding="utf-8",
        )
        return {
            "runner": type("Runner", (), {"model": DummyModel()})(),
            "tf_data": {"test_ds": object()},
            "result": {
                "timestamp": "2026-04-08 10:00:00",
                "evaluation_split": "test",
                "experiment_dir": str(output_paths["models"]),
                "training_time_seconds": 1.23,
                "total_epochs": 2,
                "head_epochs": 1,
                "ft_epochs": 1,
                "test_accuracy": 0.9,
                "test_precision": 0.91,
                "test_recall": 0.89,
                "test_f1": 0.9,
                "test_balanced_accuracy": 0.9,
                "test_roc_auc": 0.95,
                "test_pr_auc": 0.94,
            },
            "model_path": model_path,
            "history_path": history_path,
        }

    def fake_export_test_predictions(model, test_dataset, image_ids, threshold, output_path):
        df = pd.DataFrame(
            {
                "image_id": ["A", "B"],
                "y_true": [1, 0],
                "y_pred": [1, 0],
                "y_score": [0.91, 0.11],
                "split": ["test", "test"],
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    def fake_save_visualizations(predictions_df, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "confusion_matrix.png").write_text("cm", encoding="utf-8")
        (output_dir / "roc_pr_curves.png").write_text("rocpr", encoding="utf-8")

    def fake_execute_notebook_report(notebook_path, repo_root, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        executed_notebook_path = output_dir / "a11_pipeline_e2e.executed.ipynb"
        executed_notebook_path.write_text("executed", encoding="utf-8")
        notebook_visualizations_dir = output_dir.parent / "notebook_visualizations"
        notebook_visualizations_dir.mkdir(parents=True, exist_ok=True)
        return {
            "executed_notebook_path": executed_notebook_path,
            "notebook_visualizations_dir": notebook_visualizations_dir,
        }

    return {
        "build_summary": build_summary,
        "ensure_output_dirs": ensure_output_dirs,
        "execute_notebook_report": fake_execute_notebook_report,
        "export_test_predictions": fake_export_test_predictions,
        "load_pipeline_config": load_pipeline_config,
        "prepare_split_data": fake_prepare_split_data,
        "run_training_pipeline": fake_run_training_pipeline,
        "save_summary_files": save_summary_files,
        "save_visualizations": fake_save_visualizations,
        "validate_input_files": validate_input_files,
    }


def test_a11_cli_help_module():
    result = subprocess.run(
        [sys.executable, "-m", "artefatos.a11_pipeline_e2e", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--skip-train" in result.stdout
    assert "--execute-notebook" in result.stdout


def test_a11_cli_help_script():
    result = subprocess.run(
        [sys.executable, "artefatos/a11_pipeline_e2e/main.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "Pipeline end-to-end do A11" in result.stdout


def test_a11_pipeline_minimal_execution(tmp_path: Path):
    import artefatos.a11_pipeline_e2e.main as a11_main

    dataset_csv, codes_json = _write_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", dataset_csv, codes_json)
    output_root = tmp_path / "run_outputs"

    split_data = {
        "image_ids_test": ["A", "B"],
        "split_meta": {
            "n_total": 4,
            "n_valid": 4,
            "n_train": 2,
            "n_val": 0,
            "n_test": 2,
        },
    }

    summary = a11_main.run_pipeline(
        argparse.Namespace(
            config=config_path,
            limit_samples=2,
            skip_train=False,
            skip_inference=False,
            output_dir=output_root,
            execute_notebook=False,
        ),
        deps=_build_runtime_deps(split_data),
    )

    summary_json_path = output_root / "metrics" / "summary.json"
    summary_csv_path = output_root / "metrics" / "summary.csv"
    predictions_path = output_root / "predictions" / "test_predictions.csv"
    confusion_matrix_path = output_root / "visualizations" / "confusion_matrix.png"
    roc_pr_path = output_root / "visualizations" / "roc_pr_curves.png"

    assert summary["evaluation_split"] == "test"
    assert summary["test_accuracy"] == 0.9
    assert summary["predictions_path"] == str(predictions_path)
    assert summary_json_path.exists()
    assert summary_csv_path.exists()
    assert (output_root / "models" / "best_model.keras").exists()
    assert (output_root / "models" / "history.json").exists()
    assert predictions_path.exists()
    assert confusion_matrix_path.exists()
    assert roc_pr_path.exists()

    summary_json = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert _expected_summary_keys().issubset(summary_json.keys())
    assert summary_json["evaluation_split"] == "test"
    assert summary_json["test_precision"] == 0.91

    with summary_csv_path.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert _expected_summary_keys().issubset(row.keys())
    assert row["evaluation_split"] == "test"


def test_a11_pipeline_skip_inference_keeps_summary_without_predictions(tmp_path: Path):
    import artefatos.a11_pipeline_e2e.main as a11_main

    dataset_csv, codes_json = _write_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", dataset_csv, codes_json)
    output_root = tmp_path / "run_outputs"

    split_data = {
        "image_ids_test": ["A", "B"],
        "split_meta": {
            "n_total": 4,
            "n_valid": 4,
            "n_train": 2,
            "n_val": 0,
            "n_test": 2,
        },
    }

    summary = a11_main.run_pipeline(
        argparse.Namespace(
            config=config_path,
            limit_samples=None,
            skip_train=False,
            skip_inference=True,
            output_dir=output_root,
            execute_notebook=False,
        ),
        deps=_build_runtime_deps(split_data),
    )

    assert summary["predictions_path"] is None
    assert (output_root / "metrics" / "summary.json").exists()
    assert not (output_root / "predictions" / "test_predictions.csv").exists()
    assert not (output_root / "visualizations" / "confusion_matrix.png").exists()


def test_a11_pipeline_missing_dataset(tmp_path: Path):
    from artefatos.a11_pipeline_e2e.src.preprocessing import (
        load_pipeline_config,
        validate_input_files,
    )

    config_path = _write_config(
        tmp_path / "config.yaml",
        tmp_path / "missing.csv",
        tmp_path / "missing_codes.json",
    )
    config = load_pipeline_config(config_path)

    with pytest.raises(FileNotFoundError):
        validate_input_files(config)


def test_a11_pipeline_execute_notebook_exports_report_paths(tmp_path: Path):
    import artefatos.a11_pipeline_e2e.main as a11_main

    dataset_csv, codes_json = _write_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", dataset_csv, codes_json)
    output_root = tmp_path / "run_outputs"

    split_data = {
        "image_ids_test": ["A", "B"],
        "split_meta": {
            "n_total": 4,
            "n_valid": 4,
            "n_train": 2,
            "n_val": 0,
            "n_test": 2,
        },
    }

    summary = a11_main.run_pipeline(
        argparse.Namespace(
            config=config_path,
            limit_samples=None,
            skip_train=False,
            skip_inference=False,
            output_dir=output_root,
            execute_notebook=True,
        ),
        deps=_build_runtime_deps(split_data),
    )

    assert summary["executed_notebook_path"].endswith("a11_pipeline_e2e.executed.ipynb")
    assert summary["notebook_visualizations_dir"].endswith("notebook_visualizations")
    assert (output_root / "notebooks" / "a11_pipeline_e2e.executed.ipynb").exists()


def test_a11_config_paths_are_resolved_from_config_location(tmp_path: Path):
    from artefatos.a11_pipeline_e2e.src.preprocessing import load_pipeline_config

    nested_dir = tmp_path / "nested" / "artifact"
    nested_dir.mkdir(parents=True)
    dataset_csv, codes_json = _write_dataset(tmp_path)
    config_path = _write_config(nested_dir / "config.yaml", dataset_csv, codes_json)

    config = load_pipeline_config(config_path)

    assert config["paths"]["outputs_metrics"] == (nested_dir / "outputs/metrics").resolve()
    assert config["paths"]["outputs_models"] == (nested_dir / "outputs/models").resolve()
    assert config["paths"]["outputs_viz"] == (nested_dir / "outputs/visualizations").resolve()
    assert config["paths"]["outputs_preds"] == (nested_dir / "outputs/predictions").resolve()


def test_a11_prepare_split_data_is_deterministic(tmp_path: Path):
    from artefatos.a11_pipeline_e2e.src.preprocessing import (
        load_pipeline_config,
        prepare_split_data,
    )

    dataset_csv, codes_json = _write_split_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", dataset_csv, codes_json)
    config = load_pipeline_config(config_path)

    first = prepare_split_data(config)
    second = prepare_split_data(config)

    assert list(first["image_ids_train"]) == list(second["image_ids_train"])
    assert list(first["image_ids_val"]) == list(second["image_ids_val"])
    assert list(first["image_ids_test"]) == list(second["image_ids_test"])
    assert first["split_meta"]["n_train"] == second["split_meta"]["n_train"]
    assert first["split_meta"]["n_val"] == second["split_meta"]["n_val"]
    assert first["split_meta"]["n_test"] == second["split_meta"]["n_test"]


def test_a11_pipeline_skip_train_requires_saved_model(tmp_path: Path):
    pytest.importorskip("tensorflow")

    from artefatos.a11_pipeline_e2e.src.preprocessing import (
        ensure_output_dirs,
        load_pipeline_config,
    )
    from artefatos.a11_pipeline_e2e.src.training import run_training_pipeline

    dataset_csv, codes_json = _write_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.yaml", dataset_csv, codes_json)
    config = load_pipeline_config(config_path)
    output_paths = ensure_output_dirs(config, tmp_path / "outputs")

    with pytest.raises(FileNotFoundError):
        run_training_pipeline(
            config=config,
            split_data={},
            output_paths=output_paths,
            skip_train=True,
        )


def test_execute_notebook_report_runs_temp_notebook(tmp_path: Path):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    pytest.importorskip("ipykernel")

    from artefatos.a11_pipeline_e2e.src.reporting import execute_notebook_report

    repo_root = tmp_path / "repo"
    notebook_dir = repo_root / "artefatos" / "a11_pipeline_e2e" / "notebooks"
    notebook_dir.mkdir(parents=True)
    notebook_path = notebook_dir / "temp_report.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                "Path('marker.txt').write_text('ok', encoding='utf-8')\n"
                "print('executed')\n"
            )
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    with notebook_path.open("w", encoding="utf-8") as file:
        nbformat.write(notebook, file)

    result = execute_notebook_report(
        notebook_path=notebook_path,
        repo_root=repo_root,
    )

    assert (repo_root / "marker.txt").exists()
    assert result["executed_notebook_path"].exists()

    with result["executed_notebook_path"].open("r", encoding="utf-8") as file:
        executed = nbformat.read(file, as_version=4)
    assert executed.cells[0]["execution_count"] is not None
