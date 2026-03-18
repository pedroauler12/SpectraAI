"""
Pipeline tf.data para treino de CNN com augmentacao e ajuste de tensores.

Decisoes tecnicas:
- Formato padrao em channels_last (N, H, W, C), compativel com Conv2D no tf.keras.
- Conversao para float32 para eficiencia e compatibilidade com kernels do TensorFlow.
- Normalizacao por canal (zscore/minmax), ajustada no treino e reaproveitada em val/test.
- Augmentacao apenas no treino para evitar contaminacao da avaliacao.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .cnn_data_prep import apply_channel_normalizer, fit_channel_normalizer


def build_data_augmentation_layer(
    *,
    flip_mode: str = "horizontal_and_vertical",
    rotation_factor: float = 0.08,
    contrast_factor: float = 0.2,
    seed: int | None = 42,
) -> tf.keras.Sequential:
    """
    Cria bloco de augmentacao com RandomFlip, RandomRotation e RandomContrast.
    """
    if rotation_factor < 0:
        raise ValueError("rotation_factor deve ser >= 0.")
    if contrast_factor < 0:
        raise ValueError("contrast_factor deve ser >= 0.")

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode=flip_mode, seed=seed, name="random_flip"),
            tf.keras.layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="reflect",
                seed=seed,
                name="random_rotation",
            ),
            tf.keras.layers.RandomContrast(
                factor=contrast_factor,
                seed=seed,
                name="random_contrast",
            ),
        ],
        name="cnn_data_augmentation",
    )


def resize_image_tensor(
    x: np.ndarray,
    *,
    size: tuple[int, int] | None = None,
    method: str = "bilinear",
    antialias: bool = True,
) -> np.ndarray:
    """
    Redimensiona um tensor 4D ``(N, H, W, C)`` para ``size`` usando ``tf.image.resize``.
    """
    x_in = np.asarray(x)
    if x_in.ndim != 4:
        raise ValueError(f"Esperado tensor 4D, recebido shape={x_in.shape}.")
    if size is None:
        return x_in.astype(np.float32, copy=False)

    target_h, target_w = (int(size[0]), int(size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError("size deve conter duas dimensoes > 0.")
    if x_in.shape[1] == target_h and x_in.shape[2] == target_w:
        return x_in.astype(np.float32, copy=False)

    resized = tf.image.resize(
        x_in.astype(np.float32, copy=False),
        size=(target_h, target_w),
        method=method,
        antialias=antialias,
    )
    return resized.numpy().astype(np.float32, copy=False)


def _normalizer_to_channels_last(normalizer: dict[str, Any]) -> dict[str, Any]:
    """
    Converte normalizador salvo em channels_first para broadcast em channels_last.
    """
    method = normalizer.get("method")
    if method not in {"zscore", "minmax"}:
        raise ValueError("Normalizador invalido: method deve ser 'zscore' ou 'minmax'.")

    data_format = normalizer.get("data_format", "channels_last")
    if data_format == "channels_last":
        return normalizer
    if data_format != "channels_first":
        raise ValueError("Normalizador invalido: data_format deve ser channels_last/channels_first.")

    out = dict(normalizer)
    out["data_format"] = "channels_last"

    if method == "zscore":
        out["mean"] = np.transpose(normalizer["mean"], (0, 2, 3, 1))
        out["std"] = np.transpose(normalizer["std"], (0, 2, 3, 1))
        return out

    out["min"] = np.transpose(normalizer["min"], (0, 2, 3, 1))
    out["span"] = np.transpose(normalizer["span"], (0, 2, 3, 1))
    return out


def adapt_cnn_input_tensor(
    x: np.ndarray,
    *,
    data_format: str = "channels_last",
    resize_to: tuple[int, int] | None = None,
    resize_method: str = "bilinear",
    resize_antialias: bool = True,
    target_channels: int | None = None,
    repeat_single_channel: bool = False,
    normalization: str = "none",
    normalizer: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """
    Ajusta tensor para CNN em channels_last + normalizacao por canal opcional.

    Aceita:
    - shape (N, H, W): adiciona canal unico.
    - shape (N, H, W, C): mantem channels_last.
    - shape (N, C, H, W): converte de channels_first para channels_last.
    """
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError("data_format deve ser 'channels_last' ou 'channels_first'.")
    if normalization not in {"none", "zscore", "minmax"}:
        raise ValueError("normalization deve ser 'none', 'zscore' ou 'minmax'.")

    x_in = np.asarray(x)
    if x_in.ndim not in {3, 4}:
        raise ValueError(f"Esperado tensor 3D/4D, recebido shape={x_in.shape}.")

    if x_in.ndim == 3:
        if data_format == "channels_first":
            x_in = x_in[:, np.newaxis, :, :]
        else:
            x_in = x_in[..., np.newaxis]

    if data_format == "channels_first":
        x_out = np.transpose(x_in, (0, 2, 3, 1))
    else:
        x_out = x_in

    x_out = x_out.astype(np.float32, copy=False)

    x_out = resize_image_tensor(
        x_out,
        size=resize_to,
        method=resize_method,
        antialias=resize_antialias,
    )

    if target_channels is not None and target_channels <= 0:
        raise ValueError("target_channels deve ser > 0.")
    if target_channels is not None and x_out.shape[-1] != target_channels:
        if repeat_single_channel and x_out.shape[-1] == 1:
            x_out = np.repeat(x_out, target_channels, axis=-1)
        else:
            raise ValueError(
                f"Numero de canais inesperado: recebido={x_out.shape[-1]}, "
                f"esperado={target_channels}."
            )

    fitted_normalizer = normalizer
    if normalization != "none":
        if fitted_normalizer is None:
            fitted_normalizer = fit_channel_normalizer(
                x_out,
                method=normalization,
                data_format="channels_last",
            )
        else:
            if fitted_normalizer.get("method") != normalization:
                raise ValueError(
                    f"Normalizador com method={fitted_normalizer.get('method')} "
                    f"nao combina com normalization={normalization}."
                )
            fitted_normalizer = _normalizer_to_channels_last(fitted_normalizer)

        x_out = apply_channel_normalizer(x_out, fitted_normalizer)

    return x_out.astype(np.float32, copy=False), fitted_normalizer


def build_tf_data_pipeline(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    batch_size: int = 32,
    training: bool = False,
    shuffle: bool = True,
    shuffle_buffer: int = 1024,
    seed: int = 42,
    augment: bool = False,
    augmentation_layer: tf.keras.layers.Layer | None = None,
    normalization: str = "none",
    normalizer: dict[str, Any] | None = None,
    data_format: str = "channels_last",
    resize_to: tuple[int, int] | None = None,
    resize_method: str = "bilinear",
    resize_antialias: bool = True,
    target_channels: int | None = None,
    repeat_single_channel: bool = False,
    drop_remainder: bool = False,
) -> tuple[tf.data.Dataset, dict[str, Any]]:
    """
    Monta Dataset tf.data com ajuste de tensor, normalizacao e augmentacao opcional.
    """
    if batch_size <= 0:
        raise ValueError("batch_size deve ser > 0.")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer deve ser > 0.")

    x_ready, fitted_normalizer = adapt_cnn_input_tensor(
        x,
        data_format=data_format,
        resize_to=resize_to,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        target_channels=target_channels,
        repeat_single_channel=repeat_single_channel,
        normalization=normalization,
        normalizer=normalizer,
    )

    n_samples = int(x_ready.shape[0])
    if y is not None:
        y_ready = np.asarray(y)
        if y_ready.shape[0] != n_samples:
            raise ValueError(
                f"Tamanho de y ({y_ready.shape[0]}) difere de tamanho de X ({n_samples})."
            )
        ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_ready, y_ready))
    else:
        ds = tf.data.Dataset.from_tensor_slices(x_ready)

    if training and shuffle:
        buffer_size = min(n_samples, shuffle_buffer)
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if training and augment:
        aug = augmentation_layer or build_data_augmentation_layer(seed=seed)
        if y is None:
            ds = ds.map(
                lambda xb: aug(xb, training=True),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda xb, yb: (aug(xb, training=True), yb),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    meta = {
        "n_samples": n_samples,
        "batch_size": int(batch_size),
        "input_shape": tuple(int(v) for v in x_ready.shape[1:]),
        "normalization": normalization,
        "normalizer": fitted_normalizer,
        "training": bool(training),
        "augment": bool(augment),
    }
    return ds, meta


def build_train_val_tf_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 32,
    normalization: str = "zscore",
    data_format: str = "channels_last",
    resize_to: tuple[int, int] | None = None,
    target_channels: int | None = None,
    augment_train: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Cria pipeline treino/validacao reutilizando normalizador do treino.
    """
    train_ds, train_meta = build_tf_data_pipeline(
        x_train,
        y_train,
        batch_size=batch_size,
        training=True,
        shuffle=True,
        seed=seed,
        augment=augment_train,
        normalization=normalization,
        data_format=data_format,
        resize_to=resize_to,
        target_channels=target_channels,
    )

    val_ds, val_meta = build_tf_data_pipeline(
        x_val,
        y_val,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
        augment=False,
        normalization=normalization,
        normalizer=train_meta["normalizer"],
        data_format=data_format,
        resize_to=resize_to,
        target_channels=target_channels,
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "normalizer": train_meta["normalizer"],
        "train_meta": train_meta,
        "val_meta": val_meta,
    }


def build_train_val_test_tf_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_size: int = 32,
    normalization: str = "zscore",
    data_format: str = "channels_last",
    resize_to: tuple[int, int] | None = None,
    target_channels: int | None = None,
    augment_train: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Cria pipeline treino/validacao/teste reutilizando normalizador do treino.
    """
    out = build_train_val_tf_data(
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=batch_size,
        normalization=normalization,
        data_format=data_format,
        resize_to=resize_to,
        target_channels=target_channels,
        augment_train=augment_train,
        seed=seed,
    )

    test_ds, test_meta = build_tf_data_pipeline(
        x_test,
        y_test,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
        augment=False,
        normalization=normalization,
        normalizer=out["normalizer"],
        data_format=data_format,
        resize_to=resize_to,
        target_channels=target_channels,
    )

    out["test_ds"] = test_ds
    out["test_meta"] = test_meta
    return out
