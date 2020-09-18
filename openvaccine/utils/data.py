from typing import List, Dict, Any, Sequence, Union, Optional

import jsonlines


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
