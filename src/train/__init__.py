"""
Módulo de treinamento e avaliação de modelos.
"""

from .train_and_evaluate import train_and_evaluate
from .callbacks import (
    create_early_stopping,
    create_model_checkpoint,
    get_training_callbacks,
)

__all__ = [
    'train_and_evaluate',
    'create_early_stopping',
    'create_model_checkpoint',
    'get_training_callbacks',
]
