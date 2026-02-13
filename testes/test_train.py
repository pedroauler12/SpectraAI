import pytest
import numpy as np


def test_train_and_evaluate_classification():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    from src.train.train_and_evaluate import train_and_evaluate

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    results = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    assert "metrics" in results
    assert "accuracy" in results["metrics"]
    assert isinstance(results["metrics"]["accuracy"], float)


def test_train_and_evaluate_regression():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    from src.train.train_and_evaluate import train_and_evaluate

    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    results = train_and_evaluate(model, X_train, y_train, X_val, y_val, task_type="regression")

    assert "metrics" in results
    assert "mse" in results["metrics"]
    assert isinstance(results["metrics"]["mse"], float)
