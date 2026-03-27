"""
Utilitarios de inferencia para modelos Keras de classificacao binaria.

Este modulo foi criado para notebooks de analise que reutilizam modelos
com saida ``sigmoid``. Diferente de ``batch_predict``, aqui a predicao
principal e tratada como probabilidade da classe positiva.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _extract_labels_from_dataset(dataset: Iterable) -> np.ndarray | None:
    """
    Extrai ``y_true`` de datasets que iteram em lotes ``(x, y)``.

    Se o dataset nao tiver labels, retorna ``None``.
    """
    labels: list[np.ndarray] = []

    for batch in dataset:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return None

        y_batch = np.asarray(batch[1]).reshape(-1)
        labels.append(y_batch)

    if not labels:
        return np.array([], dtype=np.int64)

    return np.concatenate(labels).astype(np.int64, copy=False)


def collect_binary_predictions(
    model,
    dataset: Iterable,
    sample_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Coleta probabilidades e rótulos verdadeiros de um modelo Keras binario.

    Parameters
    ----------
    model : Any
        Modelo com metodo ``predict(dataset, verbose=0)``.
    dataset : Iterable
        Dataset iteravel, tipicamente ``tf.data.Dataset``. Quando seus lotes
        seguem o formato ``(x, y)``, ``y_true`` e extraido automaticamente.
    sample_ids : Sequence[str], optional
        Identificadores das amostras. Se ``None``, usa indices sequenciais.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ``sample_id``, ``y_true``, ``prob_pos`` e
        ``prob_neg``.
    """
    raw_pred = model.predict(dataset, verbose=0)
    prob_pos = np.asarray(raw_pred, dtype=float).reshape(-1)

    if np.any(prob_pos < -1e-8) or np.any(prob_pos > 1 + 1e-8):
        raise ValueError(
            "As predições do modelo devem representar probabilidades em [0, 1]."
        )

    prob_pos = np.clip(prob_pos, 0.0, 1.0)
    n_samples = int(prob_pos.shape[0])

    y_true = _extract_labels_from_dataset(dataset)
    if y_true is not None and y_true.shape[0] != n_samples:
        raise ValueError(
            f"Tamanho de y_true ({y_true.shape[0]}) difere do numero de predições ({n_samples})."
        )

    if sample_ids is None:
        sample_ids = [str(i) for i in range(n_samples)]
    elif len(sample_ids) != n_samples:
        raise ValueError(
            f"Tamanho de sample_ids ({len(sample_ids)}) difere do numero de predições ({n_samples})."
        )

    data = {
        "sample_id": list(sample_ids),
        "y_true": y_true if y_true is not None else np.full(n_samples, np.nan),
        "prob_pos": prob_pos,
        "prob_neg": 1.0 - prob_pos,
    }
    return pd.DataFrame(data)
