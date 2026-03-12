"""
Controles globais de reprodutibilidade para experimentos de ML.

Uso:
    from src.reprodutibilidade import set_global_seed, stratified_split

    set_global_seed(42)

    X_train, X_test, y_train, y_test = stratified_split(X, y)
"""

import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Constante global – seed padrão do projeto
# ---------------------------------------------------------------------------
SEED: int = 42


# ---------------------------------------------------------------------------
# 1. Fixar seeds globais (Python, NumPy, TensorFlow)
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """
    Define seeds globais e controles de determinismo.

    Cobre Python stdlib, NumPy e TensorFlow (se instalado).

    Parameters
    ----------
    seed : int
        Seed principal do experimento.
    deterministic : bool
        Se True, reduz variações por paralelismo (threads = 1) e ativa
        operações determinísticas no TensorFlow.
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow (import condicional para não quebrar quem não usa)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    if deterministic:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    print(f"[Reprodutibilidade] Seed definida: {seed}")


# ---------------------------------------------------------------------------
# 2. Split estratificado reutilizável
# ---------------------------------------------------------------------------
def stratified_split(
    X,
    y,
    test_size: float = 0.2,
    seed: int = SEED,
    groups=None,
):
    """
    Split estratificado para garantir comparação justa entre modelos.

    Quando *groups* é fornecido (ex.: image_id), usa GroupShuffleSplit
    para evitar vazamento de dados (pixels da mesma imagem só aparecem
    em um dos conjuntos). Caso contrário, usa StratifiedShuffleSplit.

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like
        Labels (deve ser categórico para estratificação).
    test_size : float, default 0.2
        Proporção para teste.
    seed : int, default SEED (42)
        Seed para reprodutibilidade.
    groups : array-like, optional
        Identificadores de grupo (ex.: image_id). Se fornecido,
        garante que todas as amostras de um grupo fiquem no mesmo
        conjunto (treino OU teste).

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    """
    from sklearn.model_selection import (
        GroupShuffleSplit,
        StratifiedShuffleSplit,
    )

    y_arr = np.asarray(y)

    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(X, y_arr, groups))
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(X, y_arr))

    X_arr = np.asarray(X) if not hasattr(X, "iloc") else X
    if hasattr(X_arr, "iloc"):
        X_train, X_test = X_arr.iloc[train_idx], X_arr.iloc[test_idx]
    else:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]

    y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    print(
        f"[Split] train={len(train_idx)} | test={len(test_idx)} "
        f"| pos_train={y_train.sum()} | pos_test={y_test.sum()}"
    )

    return X_train, X_test, y_train, y_test
