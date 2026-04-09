from pathlib import Path

import pytest


def test_predict_chip_array_basic(tmp_path: Path):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    from src.inference.transfer_geo_inference import (
        TransferInferenceBundle,
        predict_chip_array,
    )

    class DummyModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.73]], dtype=np.float32)

    bundle = TransferInferenceBundle(
        project_root=tmp_path,
        model=DummyModel(),
        normalizer={
            "method": "zscore",
            "data_format": "channels_last",
            "mean": np.zeros((1, 1, 1, 9), dtype=np.float32),
            "std": np.ones((1, 1, 1, 9), dtype=np.float32),
        },
        target_size=(160, 160),
        target_channels=9,
        normalization="zscore",
        class_names=("Negativo", "Positivo"),
        decision_threshold=0.5,
        seed=42,
        dataset_csv=tmp_path / "pixels_dataset.csv",
        extracted_codes_json=tmp_path / "extracted_codes.json",
        model_path=tmp_path / "best_model.keras",
    )

    chip = np.random.rand(128, 128, 9).astype(np.float32)
    result = predict_chip_array(
        bundle,
        chip,
        lat=-15.0,
        lon=-47.0,
        preview_png_path=tmp_path / "preview.png",
    )

    assert 0.0 <= result.prob_pos <= 1.0
    assert result.pred_class == 1
    assert result.pred_label == "Positivo"
    assert result.chip_shape == (128, 128, 9)
    assert result.preview_png_path is not None
    assert Path(result.preview_png_path).exists()
    assert isinstance(pd.DataFrame([result.to_dict()]), pd.DataFrame)


def test_predict_point_from_earthdata_with_mocks(tmp_path: Path, monkeypatch):
    np = pytest.importorskip("numpy")

    from src.inference import transfer_geo_inference as geo
    from src.inference.transfer_geo_inference import (
        TransferInferenceBundle,
        predict_point_from_earthdata,
    )

    class DummyModel:
        def predict(self, batch, verbose=0):
            return np.array([[0.61]], dtype=np.float32)

    bundle = TransferInferenceBundle(
        project_root=tmp_path,
        model=DummyModel(),
        normalizer={
            "method": "zscore",
            "data_format": "channels_last",
            "mean": np.zeros((1, 1, 1, 9), dtype=np.float32),
            "std": np.ones((1, 1, 1, 9), dtype=np.float32),
        },
        target_size=(160, 160),
        target_channels=9,
        normalization="zscore",
        class_names=("Negativo", "Positivo"),
        decision_threshold=0.5,
        seed=42,
        dataset_csv=tmp_path / "pixels_dataset.csv",
        extracted_codes_json=tmp_path / "extracted_codes.json",
        model_path=tmp_path / "best_model.keras",
    )

    class FakeGranule:
        umm = {
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2006-05-17T00:00:00.000Z",
                }
            },
            "AdditionalAttributes": [
                {"Name": "CloudCover", "Values": ["12.5"]},
            ],
        }
        meta = {"native-id": "ASTER_FAKE_GRANULE"}

    chip_multiband = np.random.rand(64, 64, 9).astype(np.float32)

    monkeypatch.setattr(geo, "search_earthdata_granules", lambda **kwargs: [FakeGranule()])

    def fake_login(_netrc_path):
        return None

    def fake_download(granule, outdir, dry_run=False):
        outdir.mkdir(parents=True, exist_ok=True)
        for band_name in geo.DEFAULT_BAND_ORDER:
            (outdir / f"FAKE_SRF_{band_name}.tif").touch()

    def fake_list_band_tifs(folder):
        return [folder / f"FAKE_SRF_{band_name}.tif" for band_name in geo.DEFAULT_BAND_ORDER]

    def fake_crop_and_stack_multiband(tif_paths, out_tif, bbox_wgs84, dst_crs="EPSG:4326", resampling=None):
        out_tif.write_bytes(b"fake")
        return out_tif

    monkeypatch.setattr("src.tiles.earthaccess_utils.login_earthdata", fake_login)
    monkeypatch.setattr("src.tiles.earthaccess_utils.download_granule", fake_download)
    monkeypatch.setattr("src.tiles.multiband.list_band_tifs", fake_list_band_tifs)
    monkeypatch.setattr("src.tiles.multiband.crop_and_stack_multiband", fake_crop_and_stack_multiband)
    monkeypatch.setattr(geo, "read_chip_tif", lambda path: chip_multiband)

    result = predict_point_from_earthdata(
        bundle,
        lat=-15.12,
        lon=-47.33,
        netrc_path=str(tmp_path / ".netrc"),
        cache_root=tmp_path / "cache",
    )

    assert result.granule_id == "ASTER_FAKE_GRANULE"
    assert result.cloud_cover == 12.5
    assert result.pred_label == "Positivo"
    assert result.chip_path is not None
    assert result.preview_png_path is not None
