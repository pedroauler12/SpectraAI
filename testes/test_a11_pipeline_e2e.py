import json
from pathlib import Path

import pytest
import yaml


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
            "metrics": ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"],
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
    codes_json.write_text(json.dumps({"positivos": ["A"], "negativos": ["B"]}), encoding="utf-8")
    return dataset_csv, codes_json


def test_a11_pipeline_minimal_execution(tmp_path: Path, monkeypatch):
    pd = pytest.importorskip("pandas")

    import argparse
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

    class DummyModel:
        pass

    def fake_prepare_split_data(config, limit_samples=None):
        return split_data

    def fake_run_training_pipeline(config, split_data, output_paths, skip_train=False):
        model_path = output_paths["models"] / "best_model.keras"
        model_path.write_text("dummy-model", encoding="utf-8")
        history_path = output_paths["models"] / "history.json"
        history_path.write_text(json.dumps({"loss": [0.5], "accuracy": [0.8]}), encoding="utf-8")
        return {
            "runner": type("Runner", (), {"model": DummyModel()})(),
            "tf_data": {"test_ds": object()},
            "result": {
                "timestamp": "2026-04-08 10:00:00",
                "experiment_dir": str(output_paths["models"]),
                "training_time_seconds": 1.23,
                "total_epochs": 2,
                "head_epochs": 1,
                "ft_epochs": 1,
                "val_accuracy": 0.9,
                "val_precision": 0.91,
                "val_recall": 0.89,
                "val_f1": 0.9,
                "val_balanced_accuracy": 0.9,
                "val_auc_roc": 0.95,
                "val_pr_auc": 0.94,
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

    monkeypatch.setattr(a11_main, "prepare_split_data", fake_prepare_split_data)
    monkeypatch.setattr(a11_main, "run_training_pipeline", fake_run_training_pipeline)
    monkeypatch.setattr(a11_main, "export_test_predictions", fake_export_test_predictions)
    monkeypatch.setattr(a11_main, "save_visualizations", fake_save_visualizations)
    monkeypatch.setattr(
        a11_main,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            limit_samples=2,
            skip_train=False,
            skip_inference=False,
            output_dir=output_root,
        ),
    )

    assert a11_main.main() == 0
    assert (output_root / "metrics" / "summary.json").exists()
    assert (output_root / "metrics" / "summary.csv").exists()
    assert (output_root / "predictions" / "test_predictions.csv").exists()
    assert (output_root / "visualizations" / "confusion_matrix.png").exists()


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


def test_a11_pipeline_skip_train_requires_saved_model(tmp_path: Path):
    from artefatos.a11_pipeline_e2e.src.preprocessing import ensure_output_dirs, load_pipeline_config
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
