import typer

import pandas as pd

from openvaccine.utils import load_jsonlines


def main(predictions_path: str, output_path: str):
    preds = load_jsonlines(predictions_path)

    kaggle_preds = []
    for pred in preds:
        for i, logits in enumerate(pred["logits"][1:-1]):
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

    kaggle_preds = pd.DataFrame(kaggle_preds)
    typer.echo(f"Saving submission to {output_path}")
    kaggle_preds.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
