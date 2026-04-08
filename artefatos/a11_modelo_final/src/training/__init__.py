"""
Treinamento: construção do modelo, callbacks, treinamento em duas fases
(head training + fine-tuning) e checkpoint do melhor modelo.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.reprodutibilidade import set_global_seed
from src.models.cnn_builder import build_transfer_model, unfreeze_backbone_layers
from src.train.callbacks import create_early_stopping, create_model_checkpoint

logger = logging.getLogger(__name__)


def run_training(
    cfg: dict,
    tf_data: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Treina o modelo de transfer learning em duas fases.

    Fase 1 — Head training (backbone congelado):
        Treina apenas o head (channel adapter + Dense) com lr maior.

    Fase 2 — Fine-tuning (últimas N camadas desbloqueadas):
        Desbloqueia camadas do backbone e treina com lr menor.

    Returns
    -------
    dict
        Chaves: ``model``, ``history_phase1``, ``history_phase2``,
        ``model_path``, ``normalizer``, ``training_time``, ``info``.
    """
    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Resolver diretório de saída
    if output_dir is None:
        artifact_root = Path(cfg.get("_artifact_root", "."))
        output_dir = artifact_root / cfg["paths"]["outputs_models"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extrair input_shape do tf_data
    input_shape = tf_data["train_meta"]["input_shape"]
    logger.info("Input shape: %s", input_shape)

    # ── Construção do modelo ────────────────────────────────────────────────
    logger.info("Construindo modelo de transfer learning (MobileNetV2) …")
    model, build_info = build_transfer_model(
        input_shape=input_shape,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        dropout_rate=model_cfg.get("dropout", 0.5),
    )
    model.summary(print_fn=logger.info)

    # ── Callbacks ───────────────────────────────────────────────────────────
    best_model_path = str(output_dir / "best_model.keras")
    callbacks_common = [
        create_early_stopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stopping_patience", 10),
        ),
        create_model_checkpoint(filepath=best_model_path, monitor="val_loss"),
    ]

    epochs = train_cfg.get("epochs", 50)
    train_ds = tf_data["train_ds"]
    val_ds = tf_data["val_ds"]

    # ── Fase 1: head training ───────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("FASE 1 — Head training (backbone congelado)")
    logger.info("═" * 60)

    t0 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_common,
        verbose=1,
    )
    time_p1 = time.time() - t0
    logger.info("Fase 1 concluída em %.1fs — melhor val_loss=%.4f",
                time_p1, min(history_p1.history["val_loss"]))

    # ── Fase 2: fine-tuning ─────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("FASE 2 — Fine-tuning (últimas 20 camadas)")
    logger.info("═" * 60)

    unfreeze_info = unfreeze_backbone_layers(
        model,
        fine_tune_last_layers=20,
        learning_rate=train_cfg.get("learning_rate", 1e-4) / 10,
    )
    logger.info("Camadas desbloqueadas: %d", unfreeze_info["n_unfrozen"])

    # Recria callbacks com caminhos atualizados
    best_model_ft_path = str(output_dir / "best_model_finetuned.keras")
    callbacks_ft = [
        create_early_stopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stopping_patience", 10),
        ),
        create_model_checkpoint(filepath=best_model_ft_path, monitor="val_loss"),
    ]

    t1 = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_ft,
        verbose=1,
    )
    time_p2 = time.time() - t1
    logger.info("Fase 2 concluída em %.1fs — melhor val_loss=%.4f",
                time_p2, min(history_p2.history["val_loss"]))

    total_time = time_p1 + time_p2

    # ── Salvar modelo final ─────────────────────────────────────────────────
    final_model_path = str(output_dir / "final_model.keras")
    model.save(final_model_path)
    logger.info("Modelo final salvo em: %s", final_model_path)

    # Salvar normalizer
    normalizer = tf_data.get("normalizer")
    if normalizer is not None:
        norm_path = output_dir / "normalizer.json"
        norm_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in normalizer.items()
        }
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(norm_serializable, f, indent=2)
        logger.info("Normalizer salvo em: %s", norm_path)

    return {
        "model": model,
        "history_phase1": history_p1.history,
        "history_phase2": history_p2.history,
        "model_path": final_model_path,
        "best_model_path": best_model_ft_path,
        "normalizer": normalizer,
        "training_time": total_time,
        "info": {**build_info, **unfreeze_info},
    }
