from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.models.cnn_data_prep import prepare_grouped_cnn_splits
from src.utils.reprodutibilidade import set_global_seed


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    config = deepcopy(raw_config)
    config["_config_path"] = config_path
    config["_config_dir"] = config_path.parent

    for key, value in config["paths"].items():
        config["paths"][key] = _resolve_path(config_path.parent, value)

    return config


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def validate_input_files(config: dict[str, Any]) -> None:
    dataset_csv = config["paths"]["dataset_csv"]
    extracted_codes_json = config["paths"]["extracted_codes_json"]

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_csv}")
    if not extracted_codes_json.exists():
        raise FileNotFoundError(f"Arquivo de rotulos nao encontrado: {extracted_codes_json}")


def ensure_output_dirs(
    config: dict[str, Any],
    output_dir_override: str | Path | None = None,
) -> dict[str, Path]:
    if output_dir_override is not None:
        root = Path(output_dir_override).resolve()
        paths = {
            "metrics": root / "metrics",
            "models": root / "models",
            "predictions": root / "predictions",
            "visualizations": root / "visualizations",
        }
    else:
        paths = {
            "metrics": config["paths"]["outputs_metrics"],
            "models": config["paths"]["outputs_models"],
            "predictions": config["paths"]["outputs_preds"],
            "visualizations": config["paths"]["outputs_viz"],
        }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    runs_dir = paths["models"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    paths["model_runs"] = runs_dir
    return paths


def prepare_split_data(
    config: dict[str, Any],
    limit_samples: int | None = None,
) -> dict[str, Any]:
    set_global_seed(int(config["seed"]))

    dataset_csv = config["paths"]["dataset_csv"]
    extracted_codes_json = config["paths"]["extracted_codes_json"]

    df = pd.read_csv(dataset_csv)
    if limit_samples is not None:
        df = df.sample(n=min(limit_samples, len(df)), random_state=int(config["seed"]))

    return prepare_grouped_cnn_splits(
        df,
        extracted_codes_path=str(extracted_codes_json),
        test_size=float(config["data"]["test_size"]),
        val_size=float(config["data"]["val_size"]),
        seed=int(config["seed"]),
    )
