import typer

import pandas as pd

from openvaccine.utils import load_jsonlines, parse_predictions


def main(predictions_path: str, output_path: str):
    preds = load_jsonlines(predictions_path)
    kaggle_preds = parse_predictions(preds)

    kaggle_preds = pd.DataFrame(kaggle_preds)
    typer.secho(f"Saving submission to {output_path}", fg="red")
    kaggle_preds.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
