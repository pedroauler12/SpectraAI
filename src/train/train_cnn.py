"""
Lógica de treinamento para modelos CNN com Keras.

Função modular que executa o treinamento com:
- optimizer: Adam
- loss: sparse_categorical_crossentropy
- callback: CSVLogger (salva histórico por época em CSV)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_cnn(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    csv_log_path: str = "outputs/training_logs/training_log.csv",
    verbose: int = 1,
    extra_callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Treina um modelo CNN com Adam, sparse_categorical_crossentropy e CSVLogger.

    O modelo é (re)compilado internamente para garantir que optimizer e loss
    estejam conforme especificado, independente de compilação anterior.

    Parameters
    ----------
    model : keras.Model
        Modelo Keras já construído (não precisa estar compilado).
    X_train : np.ndarray
        Tensor 4D de features (N, H, W, C).
    y_train : np.ndarray
        Labels inteiros (N,) — índices de classe (0, 1, 2, ...).
    epochs : int
        Número de épocas de treinamento.
    batch_size : int
        Tamanho do batch.
    learning_rate : float
        Taxa de aprendizado do Adam.
    validation_split : float
        Fração dos dados de treino usada como validação.
    csv_log_path : str
        Caminho do arquivo CSV onde o CSVLogger salva o histórico por época.
    verbose : int
        0=silencioso, 1=barra de progresso, 2=uma linha por época.
    extra_callbacks : list, optional
        Callbacks adicionais além do CSVLogger.

    Returns
    -------
    Dict[str, Any]
        Dicionário com:
        - 'model': modelo treinado
        - 'history': dict do histórico (loss, accuracy, etc. por época)
        - 'csv_log_path': caminho do CSV gerado
        - 'training_time': tempo de treinamento em segundos
    """
    # Garantir diretório do CSV
    csv_path = Path(csv_log_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INÍCIO DO TREINAMENTO CNN")
    logger.info("=" * 60)

    # Compilar modelo com optimizer e loss especificados
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("[Compilação] optimizer=Adam(lr=%s), loss=sparse_categorical_crossentropy", learning_rate)
    logger.info("[Dados] X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)
    logger.info("[Dados] Classes únicas: %s", np.unique(y_train))
    logger.info("[Dados] Distribuição de classes: %s", dict(zip(*np.unique(y_train, return_counts=True))))
    logger.info("[Hiperparâmetros] epochs=%d, batch_size=%d, validation_split=%.2f", epochs, batch_size, validation_split)
    logger.info("[Modelo] Total de parâmetros: %d", model.count_params())

    # Callbacks
    callbacks = [CSVLogger(str(csv_path), separator=",", append=False)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    logger.info("[Callback] CSVLogger -> %s", csv_path)

    # Treinamento
    logger.info("[Treino] Iniciando model.fit()...")
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
    )
    training_time = time.time() - start

    # Resumo final
    logger.info("=" * 60)
    logger.info("RESULTADO DO TREINAMENTO")
    logger.info("=" * 60)
    logger.info("[Tempo] Duração total: %.2fs", training_time)
    logger.info("[Época final] train_loss=%.4f, train_acc=%.4f",
                history.history["loss"][-1], history.history["accuracy"][-1])
    logger.info("[Época final] val_loss=%.4f, val_acc=%.4f",
                history.history["val_loss"][-1], history.history["val_accuracy"][-1])
    logger.info("[Melhor val_loss] %.4f (época %d)",
                min(history.history["val_loss"]),
                int(np.argmin(history.history["val_loss"])) + 1)
    logger.info("[Melhor val_acc] %.4f (época %d)",
                max(history.history["val_accuracy"]),
                int(np.argmax(history.history["val_accuracy"])) + 1)
    logger.info("=" * 60)

    return {
        "model": model,
        "history": history.history,
        "csv_log_path": str(csv_path),
        "training_time": training_time,
    }


if __name__ == "__main__":
    """
    Teste rápido com dados sintéticos para validar o pipeline de treino.
    Gera imagens aleatórias 16x16 com 3 canais e 4 classes.
    """
    from tensorflow.keras import layers, Sequential

    NUM_SAMPLES = 200
    IMG_SIZE = 16
    NUM_CHANNELS = 3
    NUM_CLASSES = 4
    EPOCHS = 5

    print("\n" + "=" * 60)
    print("TESTE: train_cnn com dados sintéticos")
    print("=" * 60)

    # Dados sintéticos
    np.random.seed(42)
    X = np.random.rand(NUM_SAMPLES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")

    # Modelo simples para teste
    model = Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
        layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    # Treinar
    result = train_cnn(
        model,
        X,
        y,
        epochs=EPOCHS,
        batch_size=16,
        learning_rate=0.001,
        validation_split=0.2,
        csv_log_path="outputs/training_logs/test_training_log.csv",
        verbose=2,
    )

    print(f"\nTempo de treino: {result['training_time']:.2f}s")
    print(f"CSV salvo em: {result['csv_log_path']}")
    print(f"Épocas registradas: {len(result['history']['loss'])}")

    # Verificar CSV gerado
    import pandas as pd
    df_log = pd.read_csv(result["csv_log_path"])
    print(f"\nConteúdo do CSV ({len(df_log)} linhas):")
    print(df_log.to_string(index=False))
