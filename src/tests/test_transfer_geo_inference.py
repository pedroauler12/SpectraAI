from pathlib import Path
import importlib

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
        model_name="Dummy",
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
        decision_threshold_name="threshold_0.5",
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
        model_name="Dummy",
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
        decision_threshold_name="threshold_0.5",
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

    for module_name, attr_name, replacement in (
        ("src.tiles.earthaccess_utils", "login_earthdata", fake_login),
        ("src.tiles.earthaccess_utils", "download_granule", fake_download),
        ("src.tiles.multiband", "list_band_tifs", fake_list_band_tifs),
        ("src.tiles.multiband", "crop_and_stack_multiband", fake_crop_and_stack_multiband),
        ("tiles.earthaccess_utils", "login_earthdata", fake_login),
        ("tiles.earthaccess_utils", "download_granule", fake_download),
        ("tiles.multiband", "list_band_tifs", fake_list_band_tifs),
        ("tiles.multiband", "crop_and_stack_multiband", fake_crop_and_stack_multiband),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        monkeypatch.setattr(module, attr_name, replacement)
    monkeypatch.setattr("earthaccess.login", lambda *args, **kwargs: None)
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


def test_load_a11_transfer_inference_bundle_uses_artifact_config(tmp_path: Path, monkeypatch):
    np = pytest.importorskip("numpy")

    from src.inference import transfer_geo_inference as geo

    project_root = tmp_path / "project"
    config_dir = project_root / "artefatos" / "a11_pipeline_e2e"
    outputs_models = config_dir / "outputs" / "models"
    data_dir = project_root / "data"
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs_models.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "pixels_dataset.csv").write_text("dummy\n1\n", encoding="utf-8")
    (data_dir / "extracted_codes.json").write_text("{}", encoding="utf-8")
    (outputs_models / "best_model.keras").write_text("fake-model", encoding="utf-8")
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "paths:",
                "  dataset_csv: ../../data/pixels_dataset.csv",
                "  extracted_codes_json: ../../data/extracted_codes.json",
                "  outputs_models: outputs/models",
                "data:",
                "  num_bands: 9",
                "  normalization_method: zscore",
                "  test_size: 0.25",
                "  val_size: 0.10",
                "model:",
                "  resize_to: [160, 160]",
                "evaluation:",
                "  threshold_default: 0.42",
            ]
        ),
        encoding="utf-8",
    )

    fake_model = object()

    def fake_prepare_grouped_cnn_splits(df, **kwargs):
        assert len(df) == 1
        assert Path(kwargs["extracted_codes_path"]).name == "extracted_codes.json"
        return {
            "X_train": np.ones((2, 16, 16, 9), dtype=np.float32),
            "shape_info": {"n_channels": 9},
        }

    def fake_fit_channel_normalizer(x_train, method, data_format):
        assert x_train.shape == (2, 16, 16, 9)
        assert method == "zscore"
        assert data_format == "channels_last"
        return {
            "method": method,
            "data_format": data_format,
            "mean": np.zeros((1, 1, 1, 9), dtype=np.float32),
            "std": np.ones((1, 1, 1, 9), dtype=np.float32),
        }

    monkeypatch.setattr(geo, "prepare_grouped_cnn_splits", fake_prepare_grouped_cnn_splits)
    monkeypatch.setattr(geo, "fit_channel_normalizer", fake_fit_channel_normalizer)
    monkeypatch.setattr(geo, "_load_keras_model", lambda path: fake_model)

    bundle = geo.load_a11_transfer_inference_bundle(project_root=project_root)

    assert bundle.model is fake_model
    assert bundle.model_name == "Pipeline Final (A11)"
    assert bundle.target_size == (160, 160)
    assert bundle.target_channels == 9
    assert bundle.normalization == "zscore"
    assert bundle.decision_threshold == pytest.approx(0.42)
    assert bundle.decision_threshold_name == "threshold_default"
    assert bundle.model_path == outputs_models / "best_model.keras"
