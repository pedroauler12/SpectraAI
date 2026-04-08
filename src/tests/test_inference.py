import numpy as np
import pytest


def test_batch_predict_basic():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from src.inference.batch_predict import batch_predict

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    results = batch_predict(model, X, batch_size=20, return_proba=True)

    assert results["labels"].shape[0] == 100
    assert results["probabilities"].shape[0] == 100
    assert results["dataframe"].shape[0] == 100


def test_batch_predict_no_proba():
    class DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    from src.inference.batch_predict import batch_predict

    X = np.zeros((10, 3))
    results = batch_predict(DummyModel(), X, return_proba=True)

    assert results["probabilities"] is None
    assert results["dataframe"].shape[0] == 10
