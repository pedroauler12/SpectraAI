"""
Módulo de inferência em lote sobre tiles de teste.
"""

from .batch_predict import batch_predict
from .keras_binary_predict import collect_binary_predictions
from .transfer_geo_inference import (
    A11_RANKING_CACHE_FILENAME,
    DEFAULT_BAND_ORDER,
    DEFAULT_CLASS_NAMES,
    PredictionResult,
    TransferInferenceBundle,
    assess_chip_quality,
    build_dataset_probability_ranking,
    build_false_color_preview,
    build_rgb_preview,
    clear_cache_dir,
    load_a11_transfer_inference_bundle,
    load_transfer_inference_bundle,
    predict_chip_array,
    predict_point_from_earthdata,
    read_chip_tif,
    resolve_project_root,
    save_preview_png,
    search_earthdata_granules,
)

__all__ = [
    "batch_predict",
    "collect_binary_predictions",
    "A11_RANKING_CACHE_FILENAME",
    "DEFAULT_BAND_ORDER",
    "DEFAULT_CLASS_NAMES",
    "PredictionResult",
    "TransferInferenceBundle",
    "assess_chip_quality",
    "build_dataset_probability_ranking",
    "build_false_color_preview",
    "build_rgb_preview",
    "clear_cache_dir",
    "load_a11_transfer_inference_bundle",
    "load_transfer_inference_bundle",
    "predict_chip_array",
    "predict_point_from_earthdata",
    "read_chip_tif",
    "resolve_project_root",
    "save_preview_png",
    "search_earthdata_granules",
]
