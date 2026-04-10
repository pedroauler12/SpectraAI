import json

import numpy as np
import pandas as pd
import pytest

from src.models.cnn_data_prep import (
    apply_channel_normalizer,
    dataframe_to_cnn_tensor,
    fit_channel_normalizer,
    get_ordered_pixel_columns,
    infer_cnn_shape,
    labels_from_extracted_codes,
    prepare_cnn_inputs,
    prepare_grouped_cnn_splits,
    stratified_group_train_val_test_split,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n_samples = 6
    n_channels = 3
    height = 4
    width = 4
    pixels_per_sample = n_channels * height * width

    data = {
        "path": [
            "chips/POS_A/chip_0.tif",
            "chips/NEG_A/chip_1.tif",
            "chips/POS_B/chip_2.tif",
            "chips/NEG_B/chip_3.tif",
            "chips/UNK/chip_4.tif",
            "chips/POS_A/chip_5.tif",
        ],
        "count": [n_channels] * n_samples,
        "height": [height] * n_samples,
        "width": [width] * n_samples,
    }

    values = rng.normal(loc=100.0, scale=20.0, size=(n_samples, pixels_per_sample))
    for idx in range(pixels_per_sample):
        data[f"pixel_{idx}"] = values[:, idx]

    return pd.DataFrame(data)


@pytest.fixture
def grouped_split_df():
    rng = np.random.default_rng(7)
    n_channels = 2
    height = 2
    width = 2
    pixels_per_sample = n_channels * height * width

    rows = []
    for prefix, label_name in (("POS", 1), ("NEG", 0)):
        for group_idx in range(5):
            image_id = f"{prefix}_{group_idx}"
            for chip_idx in range(2):
                row = {
                    "path": f"chips/{image_id}/chip_{chip_idx}.tif",
                    "count": n_channels,
                    "height": height,
                    "width": width,
                }
                values = rng.normal(loc=label_name * 10.0, scale=1.0, size=pixels_per_sample)
                for pixel_idx, value in enumerate(values):
                    row[f"pixel_{pixel_idx}"] = value
                rows.append(row)

    unknown = {
        "path": "chips/UNK/chip_0.tif",
        "count": n_channels,
        "height": height,
        "width": width,
    }
    for pixel_idx, value in enumerate(rng.normal(size=pixels_per_sample)):
        unknown[f"pixel_{pixel_idx}"] = value
    rows.append(unknown)

    return pd.DataFrame(rows)


def test_get_ordered_pixel_columns_numeric_order():
    df = pd.DataFrame(
        {
            "pixel_10": [1],
            "pixel_2": [2],
            "pixel_1": [3],
        }
    )
    cols = get_ordered_pixel_columns(df)
    assert cols == ["pixel_1", "pixel_2", "pixel_10"]


def test_infer_cnn_shape_from_metadata(sample_df):
    shape = infer_cnn_shape(sample_df)
    assert shape["n_channels"] == 3
    assert shape["height"] == 4
    assert shape["width"] == 4
    assert shape["pixels_per_channel"] == 16


def test_dataframe_to_cnn_tensor_channels_last(sample_df):
    x, meta = dataframe_to_cnn_tensor(sample_df, data_format="channels_last")
    assert x.shape == (6, 4, 4, 3)
    assert meta["data_format"] == "channels_last"


def test_dataframe_to_cnn_tensor_channels_first(sample_df):
    x, meta = dataframe_to_cnn_tensor(sample_df, data_format="channels_first")
    assert x.shape == (6, 3, 4, 4)
    assert meta["data_format"] == "channels_first"


def test_labels_from_extracted_codes(tmp_path, sample_df):
    codes_path = tmp_path / "extracted_codes.json"
    codes = {
        "positivos": ["POS_A", "POS_B"],
        "negativos": ["NEG_A", "NEG_B"],
    }
    codes_path.write_text(json.dumps(codes), encoding="utf-8")

    labels, image_ids = labels_from_extracted_codes(sample_df["path"], codes_path)
    assert labels.tolist() == [1, 0, 1, 0, -1, 1]
    assert image_ids.tolist()[0] == "POS_A"
    assert image_ids.tolist()[4] == ""


def test_fit_apply_zscore_normalization(sample_df):
    x, _ = dataframe_to_cnn_tensor(sample_df, data_format="channels_last")
    norm = fit_channel_normalizer(x, method="zscore", data_format="channels_last")
    x_norm = apply_channel_normalizer(x, norm)

    means = x_norm.mean(axis=(0, 1, 2))
    stds = x_norm.std(axis=(0, 1, 2))
    assert np.all(np.abs(means) < 1e-5)
    assert np.all(np.abs(stds - 1.0) < 1e-4)


def test_fit_apply_minmax_normalization(sample_df):
    x, _ = dataframe_to_cnn_tensor(sample_df, data_format="channels_last")
    norm = fit_channel_normalizer(x, method="minmax", data_format="channels_last")
    x_norm = apply_channel_normalizer(x, norm)

    assert np.min(x_norm) >= 0.0
    assert np.max(x_norm) <= 1.0


def test_prepare_cnn_inputs_with_codes_and_drop_invalid(tmp_path, sample_df):
    codes_path = tmp_path / "extracted_codes.json"
    codes = {
        "positivos": ["POS_A", "POS_B"],
        "negativos": ["NEG_A", "NEG_B"],
    }
    codes_path.write_text(json.dumps(codes), encoding="utf-8")

    out = prepare_cnn_inputs(
        sample_df,
        extracted_codes_path=codes_path,
        normalization="zscore",
        data_format="channels_last",
        drop_invalid_labels=True,
    )

    assert out["X"].shape[0] == 5
    assert out["y"].shape[0] == 5
    assert -1 not in out["y"]
    assert out["normalizer"] is not None


def test_prepare_cnn_inputs_reuses_normalizer(sample_df):
    df_train = sample_df.iloc[:4].copy()
    df_val = sample_df.iloc[4:].copy()
    y_train = np.array([1, 0, 1, 0], dtype=np.int64)
    y_val = np.array([0, 1], dtype=np.int64)

    train_out = prepare_cnn_inputs(
        df_train,
        labels=y_train,
        normalization="zscore",
        data_format="channels_first",
    )
    val_out = prepare_cnn_inputs(
        df_val,
        labels=y_val,
        normalization="zscore",
        normalizer=train_out["normalizer"],
        data_format="channels_first",
    )

    assert train_out["X"].shape == (4, 3, 4, 4)
    assert val_out["X"].shape == (2, 3, 4, 4)
    assert val_out["normalizer"] == train_out["normalizer"]


def test_stratified_group_train_val_test_split_has_disjoint_groups(tmp_path, grouped_split_df):
    codes_path = tmp_path / "extracted_codes.json"
    codes_path.write_text(
        json.dumps(
            {
                "positivos": [f"POS_{idx}" for idx in range(5)],
                "negativos": [f"NEG_{idx}" for idx in range(5)],
            }
        ),
        encoding="utf-8",
    )

    labels, image_ids = labels_from_extracted_codes(grouped_split_df["path"], codes_path)
    valid_mask = labels != -1

    out = stratified_group_train_val_test_split(
        image_ids[valid_mask],
        labels[valid_mask],
        test_size=0.2,
        val_size=0.2,
        seed=42,
    )

    train_ids = set(out["train_ids"].tolist())
    val_ids = set(out["val_ids"].tolist())
    test_ids = set(out["test_ids"].tolist())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 10
    assert len(out["train_idx"]) + len(out["val_idx"]) + len(out["test_idx"]) == int(valid_mask.sum())


def test_prepare_grouped_cnn_splits_builds_tensor_partitions(tmp_path, grouped_split_df):
    codes_path = tmp_path / "extracted_codes.json"
    codes_path.write_text(
        json.dumps(
            {
                "positivos": [f"POS_{idx}" for idx in range(5)],
                "negativos": [f"NEG_{idx}" for idx in range(5)],
            }
        ),
        encoding="utf-8",
    )

    out = prepare_grouped_cnn_splits(
        grouped_split_df,
        extracted_codes_path=codes_path,
        data_format="channels_last",
        test_size=0.2,
        val_size=0.2,
        seed=42,
    )

    assert out["X_train"].ndim == 4
    assert out["X_val"].ndim == 4
    assert out["X_test"].ndim == 4
    assert out["X_train"].shape[-1] == 2
    assert out["X_train"].shape[1:3] == (2, 2)
    assert out["split_meta"]["n_valid"] == 20
    assert out["split_meta"]["n_train"] + out["split_meta"]["n_val"] + out["split_meta"]["n_test"] == 20
    assert set(out["image_ids_train"]).isdisjoint(set(out["image_ids_val"]))
    assert set(out["image_ids_train"]).isdisjoint(set(out["image_ids_test"]))
    assert set(out["image_ids_val"]).isdisjoint(set(out["image_ids_test"]))
