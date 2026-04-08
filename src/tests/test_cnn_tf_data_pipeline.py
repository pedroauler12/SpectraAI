import numpy as np
import tensorflow as tf

from src.models.cnn_tf_data_pipeline import (
    adapt_cnn_input_tensor,
    build_data_augmentation_layer,
    build_tf_data_pipeline,
    build_train_val_test_tf_data,
    build_train_val_tf_data,
    resize_image_tensor,
)


def _make_sample_images(n: int = 8, h: int = 16, w: int = 16, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.normal(loc=100.0, scale=20.0, size=(n, h, w, c)).astype(np.float32)


def test_build_data_augmentation_layer_contains_required_transforms():
    layer = build_data_augmentation_layer(seed=7)
    layer_types = [type(l).__name__ for l in layer.layers]

    assert "RandomFlip" in layer_types
    assert "RandomRotation" in layer_types
    assert "RandomContrast" in layer_types


def test_adapt_cnn_input_tensor_converts_channels_first_to_last():
    x = np.zeros((4, 3, 8, 8), dtype=np.float64)
    x_out, norm = adapt_cnn_input_tensor(x, data_format="channels_first", normalization="none")

    assert x_out.shape == (4, 8, 8, 3)
    assert x_out.dtype == np.float32
    assert norm is None


def test_adapt_cnn_input_tensor_adds_channel_dim_for_3d_input():
    x = np.ones((5, 12, 10), dtype=np.float32)
    x_out, _ = adapt_cnn_input_tensor(x, data_format="channels_last", normalization="none")

    assert x_out.shape == (5, 12, 10, 1)


def test_resize_image_tensor_changes_spatial_resolution():
    x = _make_sample_images(n=4, h=12, w=10, c=3)
    x_out = resize_image_tensor(x, size=(20, 16))

    assert x_out.shape == (4, 20, 16, 3)
    assert x_out.dtype == np.float32


def test_build_tf_data_pipeline_train_with_augmentation():
    x = _make_sample_images(n=10, h=12, w=12, c=3)
    y = np.array([0, 1] * 5, dtype=np.int64)
    ds, meta = build_tf_data_pipeline(
        x,
        y,
        batch_size=4,
        training=True,
        augment=True,
        normalization="zscore",
        target_channels=3,
        seed=99,
    )

    xb, yb = next(iter(ds))
    assert isinstance(xb, tf.Tensor)
    assert isinstance(yb, tf.Tensor)
    assert xb.shape == (4, 12, 12, 3)
    assert yb.shape == (4,)
    assert xb.dtype == tf.float32
    assert meta["normalizer"] is not None
    assert meta["input_shape"] == (12, 12, 3)


def test_build_train_val_tf_data_reuses_train_normalizer():
    x_train = _make_sample_images(n=12, h=10, w=10, c=2)
    y_train = np.array([0, 1] * 6, dtype=np.int64)
    x_val = _make_sample_images(n=6, h=10, w=10, c=2) + 50.0
    y_val = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    out = build_train_val_tf_data(
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=3,
        normalization="zscore",
        target_channels=2,
        augment_train=False,
    )

    assert out["normalizer"] is out["train_meta"]["normalizer"]
    assert out["val_meta"]["normalizer"] is out["train_meta"]["normalizer"]

    xb_train, yb_train = next(iter(out["train_ds"]))
    xb_val, yb_val = next(iter(out["val_ds"]))

    assert xb_train.shape == (3, 10, 10, 2)
    assert yb_train.shape == (3,)
    assert xb_val.shape == (3, 10, 10, 2)
    assert yb_val.shape == (3,)


def test_build_train_val_test_tf_data_resizes_and_reuses_normalizer():
    x_train = _make_sample_images(n=12, h=10, w=10, c=2)
    y_train = np.array([0, 1] * 6, dtype=np.int64)
    x_val = _make_sample_images(n=6, h=10, w=10, c=2)
    y_val = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    x_test = _make_sample_images(n=4, h=10, w=10, c=2)
    y_test = np.array([0, 1, 0, 1], dtype=np.int64)

    out = build_train_val_test_tf_data(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=2,
        normalization="zscore",
        resize_to=(14, 14),
        target_channels=2,
        augment_train=False,
    )

    assert out["test_meta"]["normalizer"] is out["train_meta"]["normalizer"]

    xb_test, yb_test = next(iter(out["test_ds"]))
    assert xb_test.shape == (2, 14, 14, 2)
    assert yb_test.shape == (2,)
