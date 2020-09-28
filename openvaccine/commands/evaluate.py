import json

import typer
import pandas as pd

from openvaccine.utils.data import load_jsonlines, parse_predictions
from openvaccine.utils.metrics import calculate_mcrmse


COLUMNS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]


def main(predictions_path: str, data_path: str, out_path: str = typer.Option(None)):
    data = load_jsonlines(data_path)
    data = pd.DataFrame(data)

    y_true = []
    for idx, row in data.iterrows():
        for i in range(row.seq_scored):
            y_true.append(
                {
                    "id_seqpos": row["seq_id"] + f"_{i}",
                    "reactivity": row.reactivity[i],
                    "deg_Mg_pH10": row.deg_Mg_pH10[i],
                    "deg_Mg_50C": row.deg_Mg_50C[i],
                    "deg_pH10": row.deg_pH10[i],
                    "deg_50C": row.deg_50C[i]
                }
            )
    y_true = pd.DataFrame(y_true)[COLUMNS]

    preds = load_jsonlines(predictions_path)
    kaggle_preds = parse_predictions(preds, seq_scored=data["seq_scored"].tolist())
    kaggle_preds = pd.DataFrame(kaggle_preds)[COLUMNS]

    mcrmse = calculate_mcrmse(y_true=y_true, y_pred=kaggle_preds)
    typer.echo(f"Evaluating {predictions_path}")
    typer.secho(f"MCRMSE = {mcrmse:.3f}", fg="green")

    if out_path is not None:
        metrics = {
            "MCRMSE": float(mcrmse)
        }

        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
