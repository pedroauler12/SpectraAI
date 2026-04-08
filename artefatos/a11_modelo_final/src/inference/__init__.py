"""
Inferência: predição em lote sobre novos dados, exportando
probabilidades, rótulos e métricas de forma padronizada.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.cnn_data_prep import apply_channel_normalizer
from src.models.cnn_tf_data_pipeline import resize_image_tensor

logger = logging.getLogger(__name__)


def run_inference(
    cfg: dict,
    model: tf.keras.Model,
    normalizer: dict | None,
    tf_data: dict[str, Any],
    splits: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Executa inferência em lote sobre o conjunto de teste e salva resultados.

    Etapas:
    1. Coleta probabilidades sobre o test_ds.
    2. Aplica threshold (ótimo por F1 ou default).
    3. Monta DataFrame com resultados por amostra.
    4. Salva CSV e resumo JSON.

    Returns
    -------
    dict
        ``predictions_df``, ``summary``, ``csv_path``, ``json_path``.
    """
    if output_dir is None:
        artifact_root = Path(cfg.get("_artifact_root", "."))
        output_dir = artifact_root / cfg["paths"]["outputs_preds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = tf_data["test_ds"]
    y_test = splits["y_test"]
    image_ids_test = splits.get("image_ids_test")

    logger.info("Executando inferência em lote (%d amostras) …", len(y_test))
    t0 = time.time()

    y_prob_batches = []
    for batch in test_ds:
        xb = batch[0] if isinstance(batch, tuple) else batch
        prob = model.predict(xb, verbose=0)
        y_prob_batches.append(prob)

    y_prob = np.concatenate(y_prob_batches, axis=0).squeeze()
    inference_time = time.time() - t0

    # Threshold ótimo via F1
    from src.utils.metrics import select_threshold_by_f1

    threshold = select_threshold_by_f1(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Montar DataFrame
    records = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": np.round(y_prob, 6),
    }
    if image_ids_test is not None:
        records["image_id"] = image_ids_test

    predictions_df = pd.DataFrame(records)

    # Salvar CSV
    csv_path = output_dir / "inference_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    logger.info("Predições salvas: %s", csv_path)

    # Resumo
    n_correct = int((y_pred == y_test).sum())
    summary = {
        "n_samples": int(len(y_test)),
        "n_correct": n_correct,
        "accuracy": round(n_correct / len(y_test), 4),
        "threshold": round(float(threshold), 4),
        "inference_time_s": round(inference_time, 2),
        "time_per_sample_ms": round(inference_time / max(len(y_test), 1) * 1000, 2),
        "class_distribution_pred": {
            str(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))
        },
    }

    json_path = output_dir / "inference_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Resumo salvo: %s", json_path)

    logger.info("═" * 50)
    logger.info("INFERÊNCIA CONCLUÍDA")
    for k, v in summary.items():
        logger.info("  %-25s %s", k, v)
    logger.info("═" * 50)

    return {
        "predictions_df": predictions_df,
        "summary": summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
