import numpy as np
import pytest

from src.models.mlp_activations import (
    map_hidden_activation_to_sklearn,
    relu,
    select_mlp_activations,
    sigmoid,
    softmax,
)


def test_relu():
    x = np.array([-2.0, 0.0, 3.5])
    y = relu(x)
    np.testing.assert_allclose(y, np.array([0.0, 0.0, 3.5]))


def test_sigmoid_bounds():
    x = np.array([-1000.0, 0.0, 1000.0])
    y = sigmoid(x)
    assert y[0] < 1e-10
    assert y[1] == pytest.approx(0.5)
    assert y[2] > 1.0 - 1e-10


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    probs = softmax(logits, axis=1)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-8)


def test_select_mlp_activations_binary():
    config = select_mlp_activations(task_type="classification", n_classes=2)
    assert config["hidden_activation"] == "relu"
    assert config["output_activation"] == "sigmoid"
    assert "binaria" in config["output_justification"].lower()


def test_select_mlp_activations_multiclass():
    config = select_mlp_activations(task_type="classification", n_classes=4)
    assert config["hidden_activation"] == "relu"
    assert config["output_activation"] == "softmax"
    assert "multiclasse" in config["output_justification"].lower()


def test_map_hidden_activation_to_sklearn():
    assert map_hidden_activation_to_sklearn("relu") == "relu"
    assert map_hidden_activation_to_sklearn("sigmoid") == "logistic"


def test_create_classifier_pipeline_mlp_auto():
    pytest.importorskip("sklearn")
    from src.models.pipeline_factory import create_classifier_pipeline

    pipeline = create_classifier_pipeline(
        "mlp",
        n_classes=3,
        hidden_layer_sizes=(16,),
        random_state=42,
        max_iter=10,
    )
    assert pipeline["model"].activation == "relu"


def test_create_classifier_pipeline_mlp_sigmoid():
    pytest.importorskip("sklearn")
    from src.models.pipeline_factory import create_classifier_pipeline

    pipeline = create_classifier_pipeline(
        "mlp",
        hidden_activation="sigmoid",
        hidden_layer_sizes=(8,),
        random_state=42,
        max_iter=10,
    )
    assert pipeline["model"].activation == "logistic"
