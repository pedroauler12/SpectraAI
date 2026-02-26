    """
Funcoes de ativacao e selecao de configuracao para MLP.

Este modulo implementa as ativacoes mais usadas no baseline deep learning
do projeto (ReLU, Sigmoid e Softmax) e disponibiliza uma selecao padronizada
de acordo com o tipo de problema.
"""

from typing import Dict

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit): max(0, x).
    """
    values = np.asarray(x, dtype=float)
    return np.maximum(0.0, values)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid estavel numericamente no intervalo (0, 1).
    """
    values = np.asarray(x, dtype=float)
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax estavel numericamente ao longo de um eixo.
    """
    values = np.asarray(x, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / denominator


def select_mlp_activations(task_type: str = "classification", n_classes: int = 2) -> Dict[str, str]:
    """
    Seleciona ativacoes da MLP com justificativa conforme o problema.

    Regras:
    - Camadas ocultas: ReLU (padrao), por convergencia mais rapida e menor
      problema de gradiente que Sigmoid em redes mais profundas.
    - Saida para classificacao binaria: Sigmoid.
    - Saida para classificacao multiclasse: Softmax.
    - Saida para regressao: Linear.
    """
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type deve ser 'classification' ou 'regression'.")

    if task_type == "classification" and n_classes < 2:
        raise ValueError("Para classificacao, n_classes deve ser >= 2.")

    config = {
        "hidden_activation": "relu",
        "hidden_justification": (
            "ReLU nas camadas ocultas acelera o treinamento e reduz saturacao "
            "de gradiente em comparacao com Sigmoid."
        ),
    }

    if task_type == "classification":
        if n_classes == 2:
            config["output_activation"] = "sigmoid"
            config["output_justification"] = (
                "Sigmoid na saida e adequada para classificacao binaria, "
                "produzindo probabilidade no intervalo [0, 1]."
            )
        else:
            config["output_activation"] = "softmax"
            config["output_justification"] = (
                "Softmax na saida e adequada para classificacao multiclasse, "
                "normalizando probabilidades para soma 1 entre as classes."
            )
    else:
        config["output_activation"] = "linear"
        config["output_justification"] = (
            "Saida linear e adequada para regressao, sem limitar o valor previsto."
        )

    return config


def map_hidden_activation_to_sklearn(activation_name: str) -> str:
    """
    Mapeia nome de ativacao da camada oculta para parametro do sklearn MLP.
    """
    mapping = {
        "relu": "relu",
        "sigmoid": "logistic",
    }

    if activation_name not in mapping:
        raise ValueError(
            "hidden_activation invalida. Use 'relu' ou 'sigmoid'."
        )

    return mapping[activation_name]
