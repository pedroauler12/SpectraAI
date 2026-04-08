from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.keras_binary_predict import collect_binary_predictions


def export_test_predictions(
    *,
    model,
    test_dataset,
    image_ids,
    threshold: float,
    output_path: str | Path,
) -> pd.DataFrame:
    predictions = collect_binary_predictions(
        model=model,
        dataset=test_dataset,
        sample_ids=[str(image_id) for image_id in image_ids],
    )
    predictions = predictions.rename(
        columns={
            "sample_id": "image_id",
            "prob_pos": "y_score",
        }
    )
    predictions["y_true"] = predictions["y_true"].astype(int)
    predictions["y_pred"] = (predictions["y_score"] >= float(threshold)).astype(int)
    predictions["split"] = "test"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions[
        ["image_id", "y_true", "y_pred", "y_score", "split"]
    ].to_csv(output_path, index=False)
    return predictions[["image_id", "y_true", "y_pred", "y_score", "split"]]
