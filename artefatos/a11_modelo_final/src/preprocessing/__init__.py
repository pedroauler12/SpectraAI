"""
Pré-processamento: carregamento do dataset, reshape para tensores CNN,
split estratificado por image_id e normalização por canal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Imports do projeto raiz — reutiliza módulos já consolidados.
from src.reprodutibilidade import set_global_seed
from src.models.cnn_data_prep import (
    prepare_grouped_cnn_splits,
)
from src.models.cnn_tf_data_pipeline import (
    build_train_val_test_tf_data,
)
from src.utils.prepare_dataset import prepare_dataset_with_groups

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path) -> dict:
    """Carrega ``config.yaml`` e resolve caminhos relativos à raiz do artefato."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_artifact_root"] = str(config_path.parent)
    return cfg


def _resolve(cfg: dict, key: str) -> Path:
    """Resolve caminho relativo ao ``_artifact_root``."""
    return Path(cfg["_artifact_root"]) / cfg["paths"][key]


# ── pipeline ───────────────────────────────────────────────────────────────
def run_preprocessing(
    cfg: dict,
    *,
    dataset_csv: str | Path,
    extracted_codes_json: str | Path,
) -> dict[str, Any]:
    """
    Executa o pipeline de pré-processamento completo.

    Etapas:
    1. Fixa seed global.
    2. Carrega CSV de pixels e gera labels a partir de ``extracted_codes.json``.
    3. Converte para tensor 4D e aplica split estratificado por ``image_id``.
    4. Constrói pipelines ``tf.data`` com normalização e augmentação.

    Returns
    -------
    dict
        Chaves: ``splits`` (tensores numpy), ``tf_data`` (datasets tf.data),
        ``cfg``, ``dataset_csv``, ``extracted_codes_json``.
    """
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    h, w = data_cfg.get("image_size", [64, 64])
    n_bands = data_cfg.get("num_bands", 9)
    test_size = data_cfg.get("test_size", 0.15)
    val_size = data_cfg.get("val_size", 0.15)

    # 1 — Carregar dataset ───────────────────────────────────────────────────
    logger.info("Carregando dataset: %s", dataset_csv)
    df = pd.read_csv(dataset_csv)
    logger.info("  Amostras brutas: %d | Colunas: %d", len(df), len(df.columns))

    # 2 — Tensor 4D + split estratificado ────────────────────────────────────
    logger.info("Convertendo pixels → tensor 4D e aplicando split por image_id …")
    splits = prepare_grouped_cnn_splits(
        df,
        extracted_codes_path=str(extracted_codes_json),
        n_channels=n_bands,
        height=h,
        width=w,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    for subset in ("train", "val", "test"):
        x_key = f"X_{subset}"
        y_key = f"y_{subset}"
        logger.info(
            "  %s: X=%s  |  classes=%s",
            subset,
            splits[x_key].shape,
            dict(zip(*np.unique(splits[y_key], return_counts=True))),
        )

    # 3 — Pipelines tf.data ──────────────────────────────────────────────────
    logger.info("Construindo pipelines tf.data (normalização + augmentação) …")
    tf_data = build_train_val_test_tf_data(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        splits["X_test"],
        splits["y_test"],
        batch_size=train_cfg.get("batch_size", 32),
        normalization="zscore",
        resize_to=None,
        target_channels=None,
        augment_train=True,
        seed=seed,
    )
    logger.info("  Input shape (pós-pipeline): %s", tf_data["train_meta"]["input_shape"])

    return {
        "splits": splits,
        "tf_data": tf_data,
        "cfg": cfg,
        "dataset_csv": str(dataset_csv),
        "extracted_codes_json": str(extracted_codes_json),
    }
