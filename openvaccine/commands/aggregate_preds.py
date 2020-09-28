from pathlib import Path

import typer
import numpy as np

from openvaccine.utils.data import write_jsonlines, load_jsonlines


def main(log_dir: str, preds_name: str = "test_preds.json"):

    final_predictions = []
    paths = list(Path(log_dir).iterdir())
    for path in paths:
        predictions = load_jsonlines(str(path / preds_name))
        for i, pred in enumerate(predictions):
            pred["logits"] = np.array(pred["logits"]) / len(paths)
            if len(final_predictions) <= i:
                final_predictions.append(pred)
            else:
                final_predictions[i]["logits"] += pred["logits"]

    for pred in final_predictions:
        pred["logits"] = pred["logits"].tolist()

    write_jsonlines(final_predictions, f"{log_dir}/{preds_name}")


if __name__ == "__main__":
    typer.run(main)
