"""
Funções para inferência em lote sobre tiles de teste.

Este módulo contém funções para rodar predições sobre conjuntos de tiles,
retornando probabilidades e rótulos preditos de forma estruturada.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def batch_predict(
    model: Any,
    X: np.ndarray,
    batch_size: Optional[int] = None,
    return_proba: bool = True,
    sample_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executa inferência em lote sobre um conjunto de tiles de teste.

    Roda predições usando um modelo treinado, retornando rótulos preditos
    e (opcionalmente) probabilidades por classe. Suporta processamento
    em mini-batches para controlar uso de memória.

    Parameters
    ----------
    model : Any
        Modelo treinado com método predict(). Para probabilidades,
        deve ter predict_proba() ou decision_function().
    X : np.ndarray
        Features do conjunto de teste, shape (n_samples, n_features).
    batch_size : int, optional
        Tamanho de cada mini-batch. Se None, processa tudo de uma vez.
    return_proba : bool, optional
        Se True, retorna probabilidades por classe (default: True).
    sample_ids : list of str, optional
        Identificadores das amostras. Se None, usa índices numéricos.

    Returns
    -------
    Dict[str, Any]
        Dicionário contendo:
        - 'labels': np.ndarray com rótulos preditos (n_samples,)
        - 'probabilities': np.ndarray com probabilidades (n_samples, n_classes)
          ou None se return_proba=False ou modelo não suporta
        - 'sample_ids': lista de identificadores das amostras
        - 'n_samples': número total de amostras processadas
        - 'inference_time': tempo total de inferência em segundos
        - 'dataframe': pd.DataFrame consolidado com resultados

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> results = batch_predict(model, X_test)
    >>> print(results['dataframe'].head())

    >>> # Com mini-batches e IDs personalizados
    >>> results = batch_predict(
    ...     model, X_test,
    ...     batch_size=256,
    ...     sample_ids=['tile_001', 'tile_002', 'tile_003']
    ... )
    >>> print(f"Predições: {results['labels']}")
    >>> print(f"Probabilidades: {results['probabilities']}")
    """
    n_samples = X.shape[0]

    if sample_ids is None:
        sample_ids = [str(i) for i in range(n_samples)]

    if len(sample_ids) != n_samples:
        raise ValueError(
            f"Tamanho de sample_ids ({len(sample_ids)}) "
            f"não corresponde ao número de amostras ({n_samples})."
        )

    logger.info(f"{'='*60}")
    logger.info(f"Iniciando inferência em lote")
    logger.info(f"Modelo: {model.__class__.__name__}")
    logger.info(f"Amostras: {n_samples}")
    logger.info(f"Features: {X.shape[1]}")
    if batch_size:
        n_batches = int(np.ceil(n_samples / batch_size))
        logger.info(f"Batch size: {batch_size} ({n_batches} batches)")
    logger.info(f"{'='*60}")

    start_time = time.time()

    if batch_size is None:
        labels, probabilities = _predict_chunk(model, X, return_proba)
    else:
        labels, probabilities = _predict_in_batches(
            model, X, batch_size, return_proba
        )

    inference_time = time.time() - start_time

    # Monta DataFrame consolidado
    df = _build_results_dataframe(sample_ids, labels, probabilities)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTADOS DA INFERENCIA EM LOTE")
    logger.info(f"{'='*60}")
    logger.info(f"Amostras processadas:............ {n_samples}")
    logger.info(f"Tempo de inferência:............. {inference_time:.2f}s")
    if n_samples > 0:
        logger.info(f"Tempo por amostra:............... {inference_time/n_samples*1000:.2f}ms")
    if probabilities is not None:
        logger.info(f"Classes detectadas:.............. {probabilities.shape[1]}")
    classes, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(classes, counts):
        logger.info(f"  Classe {cls}:..................... {cnt} ({cnt/n_samples*100:.1f}%)")
    logger.info(f"{'='*60}\n")

    results = {
        'labels': labels,
        'probabilities': probabilities,
        'sample_ids': sample_ids,
        'n_samples': n_samples,
        'inference_time': inference_time,
        'dataframe': df
    }

    return results


def _predict_chunk(
    model: Any,
    X: np.ndarray,
    return_proba: bool
) -> tuple:
    """
    Executa predição sobre um único bloco de dados.

    Parameters
    ----------
    model : Any
        Modelo treinado.
    X : np.ndarray
        Features, shape (n_samples, n_features).
    return_proba : bool
        Se True, tenta extrair probabilidades.

    Returns
    -------
    tuple
        (labels, probabilities) onde probabilities pode ser None.
    """
    try:
        labels = model.predict(X)
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise

    probabilities = None
    if return_proba:
        probabilities = _extract_probabilities(model, X)

    return labels, probabilities


def _predict_in_batches(
    model: Any,
    X: np.ndarray,
    batch_size: int,
    return_proba: bool
) -> tuple:
    """
    Executa predição dividindo os dados em mini-batches.

    Parameters
    ----------
    model : Any
        Modelo treinado.
    X : np.ndarray
        Features, shape (n_samples, n_features).
    batch_size : int
        Número de amostras por batch.
    return_proba : bool
        Se True, tenta extrair probabilidades.

    Returns
    -------
    tuple
        (labels, probabilities) concatenados de todos os batches.
    """
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    all_labels = []
    all_probas = []

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]

        logger.info(f"  Batch {i+1}/{n_batches} (amostras {start}-{end-1})")

        batch_labels, batch_proba = _predict_chunk(model, X_batch, return_proba)
        all_labels.append(batch_labels)

        if batch_proba is not None:
            all_probas.append(batch_proba)

    labels = np.concatenate(all_labels)
    probabilities = np.vstack(all_probas) if all_probas else None

    return labels, probabilities


def _extract_probabilities(
    model: Any,
    X: np.ndarray
) -> Optional[np.ndarray]:
    """
    Extrai probabilidades de um modelo, se disponível.

    Tenta predict_proba() primeiro, depois decision_function()
    como fallback.

    Parameters
    ----------
    model : Any
        Modelo treinado.
    X : np.ndarray
        Features de entrada.

    Returns
    -------
    np.ndarray or None
        Probabilidades por classe ou None se não disponível.
    """
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception as e:
            logger.warning(f"predict_proba falhou: {str(e)}")

    if hasattr(model, 'decision_function'):
        try:
            scores = model.decision_function(X)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            return scores
        except Exception as e:
            logger.warning(f"decision_function falhou: {str(e)}")

    logger.warning("Modelo não suporta probabilidades. Retornando None.")
    return None


def _build_results_dataframe(
    sample_ids: List[str],
    labels: np.ndarray,
    probabilities: Optional[np.ndarray]
) -> pd.DataFrame:
    """
    Constrói DataFrame consolidado com resultados da inferência.

    Parameters
    ----------
    sample_ids : list of str
        Identificadores das amostras.
    labels : np.ndarray
        Rótulos preditos.
    probabilities : np.ndarray or None
        Probabilidades por classe.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: sample_id, predicted_label, e
        proba_class_N para cada classe (se disponível).
    """
    data = {
        'sample_id': sample_ids,
        'predicted_label': labels
    }

    if probabilities is not None:
        n_classes = probabilities.shape[1]
        for c in range(n_classes):
            data[f'proba_class_{c}'] = probabilities[:, c]

    return pd.DataFrame(data)


if __name__ == "__main__":
    """
    Bloco de teste da função batch_predict.

    Este exemplo demonstra o uso da função com dados sintéticos
    para classificação binária.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "="*70)
    print("TESTE: INFERENCIA EM LOTE - CLASSIFICACAO BINARIA")
    print("="*70 + "\n")

    # Gera dados sintéticos
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Treina modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Teste 1: Inferência simples
    print("\n--- Teste 1: Inferência em bloco único ---\n")
    results = batch_predict(model, X_test)
    print(f"Labels shape: {results['labels'].shape}")
    print(f"Probabilities shape: {results['probabilities'].shape}")
    print(f"\nDataFrame head:\n{results['dataframe'].head()}")

    # Teste 2: Com mini-batches e IDs customizados
    print("\n--- Teste 2: Com mini-batches ---\n")
    ids = [f"tile_{i:04d}" for i in range(X_test.shape[0])]
    results_batch = batch_predict(
        model, X_test,
        batch_size=50,
        sample_ids=ids
    )
    print(f"Labels shape: {results_batch['labels'].shape}")
    print(f"Tempo total: {results_batch['inference_time']:.4f}s")
    print(f"\nDataFrame head:\n{results_batch['dataframe'].head()}")

    # Teste 3: Sem probabilidades
    print("\n--- Teste 3: Sem probabilidades ---\n")
    results_no_proba = batch_predict(
        model, X_test,
        return_proba=False
    )
    print(f"Probabilities: {results_no_proba['probabilities']}")
    print(f"\nDataFrame head:\n{results_no_proba['dataframe'].head()}")
