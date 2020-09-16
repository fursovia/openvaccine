from typing import List, Dict, Any, Sequence, Union, Optional

import jsonlines
import pandas as pd


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)


def parse_predictions(
        predictions: List[Dict[str, Any]],
        seq_scored: Optional[List[int]] = None,
) -> List[Dict[str, Union[str, float]]]:
    kaggle_preds = []
    for j, pred in enumerate(predictions):
        logits = pred["logits"][1:-1]
        if seq_scored is not None:
            logits = logits[:seq_scored[j]]

        for i, logits in enumerate(logits):
            kaggle_preds.append(
                {
                    "id_seqpos": pred["seq_id"] + f"_{i}",
                    "reactivity": logits[0],
                    "deg_Mg_pH10": logits[1],
                    "deg_Mg_50C": logits[2],
                    "deg_pH10": logits[3],
                    "deg_50C": logits[4]
                }
            )
    return kaggle_preds


def calculate_mcrmse(y_true: pd.DataFrame, y_pred: pd.DataFrame, calculate_on_scored: bool = True) -> float:
    y_true = y_true.values
    y_pred = y_pred.values

    crmse = []
    num_cols = 3 if calculate_on_scored else 5
    for j in range(num_cols):
        mse = ((y_true[:, j] - y_pred[:, j]) ** 2).mean()
        rmse = mse ** (1 / 2)
        crmse.append(rmse)

    return sum(crmse) / len(crmse)
