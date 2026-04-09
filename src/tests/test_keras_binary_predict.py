import pytest


def test_collect_binary_predictions_basic():
    pd = pytest.importorskip("pandas")

    from src.inference.keras_binary_predict import collect_binary_predictions

    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def numpy(self):
            return self._values

        def __array__(self, dtype=None):
            return self._values

    class DummyModel:
        def predict(self, dataset, verbose=0):
            return [[0.2], [0.8], [0.6], [0.1]]

    dataset = [
        ("x_batch_1", FakeTensor([0, 1])),
        ("x_batch_2", FakeTensor([1, 0])),
    ]

    df = collect_binary_predictions(
        DummyModel(),
        dataset,
        sample_ids=["a", "b", "c", "d"],
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["sample_id", "y_true", "prob_pos", "prob_neg"]
    assert df.shape == (4, 4)
    assert df["prob_pos"].between(0, 1).all()
    assert df["prob_neg"].between(0, 1).all()
    assert ((df["prob_pos"] + df["prob_neg"]) - 1.0).abs().max() < 1e-9


def test_collect_binary_predictions_without_labels():
    pytest.importorskip("pandas")

    from src.inference.keras_binary_predict import collect_binary_predictions

    class DummyModel:
        def predict(self, dataset, verbose=0):
            return [0.25, 0.75]

    df = collect_binary_predictions(DummyModel(), [1, 2])

    assert df.shape == (2, 4)
    assert df["y_true"].isna().all()
