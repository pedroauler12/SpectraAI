"""
Callbacks reutilizáveis para treinamento de modelos Keras.

Este módulo centraliza a criação de callbacks de EarlyStopping e
ModelCheckpoint, garantindo configuração padronizada em todos os
experimentos do projeto.
"""

import logging
import os
from typing import List, Optional

from tensorflow.keras.callbacks import (  # type: ignore[import-untyped]
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)

logger = logging.getLogger(__name__)


def create_early_stopping(
    monitor: str = "val_loss",
    patience: int = 10,
    min_delta: float = 1e-4,
    restore_best_weights: bool = True,
    verbose: int = 1,
) -> EarlyStopping:
    """
    Cria callback de EarlyStopping para interromper o treino quando a
    métrica monitorada parar de melhorar.

    Parameters
    ----------
    monitor : str, optional
        Métrica a ser monitorada (default: ``"val_loss"``).
    patience : int, optional
        Número de épocas sem melhoria antes de parar (default: 10).
    min_delta : float, optional
        Variação mínima para qualificar como melhoria (default: 1e-4).
    restore_best_weights : bool, optional
        Se ``True``, restaura os pesos da melhor época ao final
        (default: ``True``).
    verbose : int, optional
        Nível de verbosidade (default: 1).

    Returns
    -------
    EarlyStopping
        Callback configurado.

    Examples
    --------
    >>> es = create_early_stopping(patience=15)
    >>> model.fit(X, y, callbacks=[es])
    """
    callback = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )
    logger.info(
        "EarlyStopping configurado — monitor=%s, patience=%d, "
        "min_delta=%s, restore_best_weights=%s",
        monitor,
        patience,
        min_delta,
        restore_best_weights,
    )
    return callback


def create_model_checkpoint(
    filepath: str = "models/best_model.keras",
    monitor: str = "val_loss",
    save_best_only: bool = True,
    verbose: int = 1,
) -> ModelCheckpoint:
    """
    Cria callback de ModelCheckpoint que salva apenas o melhor modelo
    no formato ``.keras``.

    O diretório de destino é criado automaticamente caso não exista.

    Parameters
    ----------
    filepath : str, optional
        Caminho do arquivo ``.keras`` a ser salvo
        (default: ``"models/best_model.keras"``).
    monitor : str, optional
        Métrica a ser monitorada (default: ``"val_loss"``).
    save_best_only : bool, optional
        Se ``True``, sobrescreve o arquivo somente quando a métrica
        melhora (default: ``True``).
    verbose : int, optional
        Nível de verbosidade (default: 1).

    Returns
    -------
    ModelCheckpoint
        Callback configurado.

    Raises
    ------
    ValueError
        Se ``filepath`` não terminar com a extensão ``.keras``.

    Examples
    --------
    >>> ckpt = create_model_checkpoint("outputs/meu_modelo.keras")
    >>> model.fit(X, y, callbacks=[ckpt])
    """
    if not filepath.endswith(".keras"):
        raise ValueError(
            f"O filepath deve ter extensão .keras, recebido: '{filepath}'"
        )

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    callback = ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        verbose=verbose,
    )
    logger.info(
        "ModelCheckpoint configurado — filepath=%s, monitor=%s, "
        "save_best_only=%s",
        filepath,
        monitor,
        save_best_only,
    )
    return callback


def get_training_callbacks(
    checkpoint_path: str = "models/best_model.keras",
    monitor: str = "val_loss",
    patience: int = 10,
    min_delta: float = 1e-4,
    extra_callbacks: Optional[List[Callback]] = None,
) -> List[Callback]:
    """
    Retorna a lista padrão de callbacks para treinamento, já incluindo
    EarlyStopping e ModelCheckpoint com configurações consistentes.

    Esta é a função principal para uso nos notebooks e scripts de
    treinamento do projeto, garantindo reprodutibilidade.

    Parameters
    ----------
    checkpoint_path : str, optional
        Caminho do arquivo ``.keras`` para salvar o melhor modelo
        (default: ``"models/best_model.keras"``).
    monitor : str, optional
        Métrica monitorada por ambos os callbacks (default: ``"val_loss"``).
    patience : int, optional
        Épocas de tolerância para o EarlyStopping (default: 10).
    min_delta : float, optional
        Melhoria mínima para o EarlyStopping (default: 1e-4).
    extra_callbacks : list of Callback, optional
        Callbacks adicionais a serem incluídos na lista.

    Returns
    -------
    list of Callback
        Lista pronta para ser passada em ``model.fit(..., callbacks=)``.

    Examples
    --------
    >>> from src.train.callbacks import get_training_callbacks
    >>> cbs = get_training_callbacks(
    ...     checkpoint_path="outputs/a03_mlp_baseline/best.keras",
    ...     patience=15,
    ... )
    >>> history = model.fit(
    ...     X_train, y_train,
    ...     validation_data=(X_val, y_val),
    ...     epochs=200,
    ...     callbacks=cbs,
    ... )
    """
    callbacks: List[Callback] = [
        create_early_stopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
        ),
        create_model_checkpoint(
            filepath=checkpoint_path,
            monitor=monitor,
        ),
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    logger.info("Lista de callbacks montada com %d item(ns).", len(callbacks))
    return callbacks


if __name__ == "__main__":
    """
    Bloco de teste das funções de callbacks.

    Treina uma MLP simples em dados sintéticos para validar o
    funcionamento de EarlyStopping e ModelCheckpoint de ponta a ponta.
    """
    import tempfile

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from tensorflow import keras  # type: ignore[import-untyped]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ── 1. Dados sintéticos ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TESTE: EarlyStopping + ModelCheckpoint com MLP Keras")
    print("=" * 60 + "\n")

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        n_classes=2,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    # ── 2. Modelo MLP mínimo ─────────────────────────────────────────
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ── 3. Callbacks via função utilitária ────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.keras")

        callbacks = get_training_callbacks(
            checkpoint_path=ckpt_path,
            patience=5,
            min_delta=1e-3,
        )

        print(f"Callbacks criados: {[type(c).__name__ for c in callbacks]}")
        print(f"Checkpoint será salvo em: {ckpt_path}\n")

        # ── 4. Treinamento ────────────────────────────────────────────
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # ── 5. Verificações ──────────────────────────────────────────
        epochs_run = len(history.history["loss"])
        best_val_loss = min(history.history["val_loss"])
        checkpoint_exists = os.path.isfile(ckpt_path)

        print("\n" + "=" * 60)
        print("RESULTADOS")
        print("=" * 60)
        print(f"Épocas executadas .... {epochs_run} (max 100)")
        print(f"Melhor val_loss ...... {best_val_loss:.6f}")
        print(f"Checkpoint salvo ..... {'✓' if checkpoint_exists else '✗'}")

        if checkpoint_exists:
            loaded = keras.models.load_model(ckpt_path)
            preds = (loaded.predict(X_val, verbose=0) > 0.5).astype(int).ravel()
            acc = np.mean(preds == y_val)
            print(f"Acurácia (modelo carregado) .. {acc:.4f}")
            print("✓ Modelo carregado e funcional!")
        else:
            print("✗ ERRO: checkpoint não foi salvo.")

        print("=" * 60 + "\n")
