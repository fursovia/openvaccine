import typer

import pandas as pd

from openvaccine.utils import load_jsonlines, parse_predictions, calculate_mcrmse

COLUMNS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]


def main(predictions_path: str, data_path: str):
    data = load_jsonlines(data_path)
    data = pd.DataFrame(data)[COLUMNS]

    preds = load_jsonlines(predictions_path)
    kaggle_preds = parse_predictions(preds)
    kaggle_preds = pd.DataFrame(kaggle_preds)[COLUMNS]

    mcrmse = calculate_mcrmse(y_true=data, y_pred=kaggle_preds)
    typer.echo(f"Evaluating {predictions_path}")
    typer.secho(f"MCRMSE = {mcrmse:.3f}", fg="green")


if __name__ == "__main__":
    typer.run(main)
