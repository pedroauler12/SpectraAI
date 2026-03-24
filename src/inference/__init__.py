"""
Módulo de inferência em lote sobre tiles de teste.
"""

from .batch_predict import batch_predict
from .keras_binary_predict import collect_binary_predictions
from .transfer_geo_inference import (
    DEFAULT_BAND_ORDER,
    DEFAULT_CLASS_NAMES,
    PredictionResult,
    TransferInferenceBundle,
    build_false_color_preview,
    clear_cache_dir,
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
    "DEFAULT_BAND_ORDER",
    "DEFAULT_CLASS_NAMES",
    "PredictionResult",
    "TransferInferenceBundle",
    "build_false_color_preview",
    "clear_cache_dir",
    "load_transfer_inference_bundle",
    "predict_chip_array",
    "predict_point_from_earthdata",
    "read_chip_tif",
    "resolve_project_root",
    "save_preview_png",
    "search_earthdata_granules",
]
