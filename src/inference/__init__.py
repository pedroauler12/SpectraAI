"""
Módulo de inferência em lote sobre tiles de teste.
"""

from .batch_predict import batch_predict
from .keras_binary_predict import collect_binary_predictions

__all__ = ["batch_predict", "collect_binary_predictions"]
