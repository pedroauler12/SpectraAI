from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorflow import keras

from src.models.transfer_experiment_runner import TransferLearningExperimentRunner
from src.reprodutibilidade import set_global_seed


def _to_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "val_accuracy": "test_accuracy",
        "val_precision": "test_precision",
        "val_recall": "test_recall",
        "val_f1": "test_f1",
        "val_balanced_accuracy": "test_balanced_accuracy",
        "val_auc_roc": "test_roc_auc",
        "val_pr_auc": "test_pr_auc",
        "val_cm_tp": "test_cm_tp",
        "val_cm_fp": "test_cm_fp",
        "val_cm_tn": "test_cm_tn",
        "val_cm_fn": "test_cm_fn",
        "val_specificity": "test_specificity",
        "val_sensitivity": "test_sensitivity",
    }
    return {
        target_key: metrics.get(source_key)
        for source_key, target_key in key_map.items()
    }


def run_training_pipeline(
    *,
    config: dict[str, Any],
    split_data: dict[str, Any],
    output_paths: dict[str, Path],
    skip_train: bool = False,
) -> dict[str, Any]:
    set_global_seed(int(config["seed"]))

    model_path = output_paths["models"] / "best_model.keras"
    history_path = output_paths["models"] / "history.json"

    if skip_train and not model_path.exists():
        raise FileNotFoundError(
            f"--skip-train exige modelo salvo em: {model_path}"
        )

    runner = _build_runner(config, output_paths)
    tf_data = runner.build_tf_data(split_data)

    if skip_train:
        runner.model = keras.models.load_model(model_path)
        result = _evaluate_loaded_model(
            runner=runner,
            tf_data=tf_data,
            config=config,
            model_path=model_path,
            history_path=history_path,
        )
        return {
            "runner": runner,
            "tf_data": tf_data,
            "result": result,
            "model_path": model_path,
            "history_path": history_path if history_path.exists() else None,
        }

    runner.create_experiment_dir()
    runner.build_model(input_shape=tf_data["train_meta"]["input_shape"])
    train_result = runner.train_two_phases(tf_data, verbose=0)
    test_metrics = _to_test_metrics(runner.evaluate_on_test(tf_data))

    runner.model.save(model_path)
    _save_history(history_path, train_result["full_history"])

    result = {
        "config_name": config["model"]["base_config_name"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_split": "test",
        "experiment_dir": str(runner.experiment_dir),
        "training_time_seconds": runner.training_time,
        "head_epochs": train_result["head_epochs"],
        "ft_epochs": train_result["ft_epochs"],
        "total_epochs": train_result["total_epochs"],
        "final_train_loss": float(train_result["full_history"]["loss"][-1]),
        "final_train_acc": float(train_result["full_history"]["accuracy"][-1]),
        "final_val_loss": float(train_result["full_history"]["val_loss"][-1]),
        "final_val_acc": float(train_result["full_history"]["val_accuracy"][-1]),
        **test_metrics,
    }

    return {
        "runner": runner,
        "tf_data": tf_data,
        "result": result,
        "model_path": model_path,
        "history_path": history_path,
    }


def _build_runner(
    config: dict[str, Any],
    output_paths: dict[str, Path],
) -> TransferLearningExperimentRunner:
    base_config_name = config["model"]["base_config_name"]
    runner = TransferLearningExperimentRunner(base_config_name)

    runner.config = {
        "model": {
            "input_shape": [
                int(config["data"]["image_size"][0]),
                int(config["data"]["image_size"][1]),
                int(config["data"]["num_bands"]),
            ],
            "num_classes": int(config["model"]["num_classes"]),
            "backbone": config["model"]["backbone"],
            "resize_to": [int(v) for v in config["model"]["resize_to"]],
            "dropout_rate": float(config["model"]["dropout_rate"]),
            "fine_tune_last_layers": int(config["model"]["fine_tune_last_layers"]),
        },
        "training": {
            "batch_size": int(config["training"]["batch_size"]),
            "head_epochs": int(config["training"]["head_epochs"]),
            "head_learning_rate": float(config["training"]["head_learning_rate"]),
            "fine_tune_epochs": int(config["training"]["fine_tune_epochs"]),
            "fine_tune_learning_rate": float(
                config["training"]["fine_tune_learning_rate"]
            ),
            "early_stopping_patience_head": int(
                config["training"]["early_stopping_patience_head"]
            ),
            "early_stopping_patience_ft": int(
                config["training"]["early_stopping_patience_ft"]
            ),
        },
        "data": {
            "dataset_path": str(config["paths"]["dataset_csv"]),
            "codes_path": str(config["paths"]["extracted_codes_json"]),
            "normalization_method": config["data"]["normalization_method"],
            "test_size": float(config["data"]["test_size"]),
            "val_size": float(config["data"]["val_size"]),
        },
        "output": {
            "models_dir": str(output_paths["model_runs"]),
            "logs_dir": str(output_paths["metrics"]),
            "save_model": False,
            "save_history": False,
        },
    }
    return runner


def _save_history(history_path: Path, history: dict[str, list[float]]) -> None:
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.items()
    }
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)


def _evaluate_loaded_model(
    *,
    runner: TransferLearningExperimentRunner,
    tf_data: dict[str, Any],
    config: dict[str, Any],
    model_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    metrics = _to_test_metrics(runner.evaluate_on_test(tf_data))
    total_epochs = None
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        total_epochs = len(history.get("loss", [])) or None

    return {
        "config_name": config["model"]["base_config_name"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_split": "test",
        "experiment_dir": str(model_path.parent),
        "training_time_seconds": None,
        "head_epochs": None,
        "ft_epochs": None,
        "total_epochs": total_epochs,
        "final_train_loss": None,
        "final_train_acc": None,
        "final_val_loss": None,
        "final_val_acc": None,
        **metrics,
    }
