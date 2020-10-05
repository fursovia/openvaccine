from pathlib import Path

import typer
import pandas as pd

COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]


def main(log_dir: str, postfix: str = "_submit.csv"):

    predictions = None
    paths = list(Path(log_dir).glob(f"*{postfix}"))
    for path in paths:
        preds = pd.read_csv(path)

        if predictions is None:
            predictions = preds[COLS].values / len(paths)
        else:
            predictions += preds[COLS].values / len(paths)

    final_preds = pd.DataFrame(
        {
            "id_seqpos": preds["id_seqpos"],
            "reactivity": predictions[:, 0],
            "deg_Mg_pH10": predictions[:, 1],
            "deg_Mg_50C": predictions[:, 2],
            "deg_pH10": predictions[:, 3],
            "deg_50C": predictions[:, 4],
        }
    )
    final_preds.to_csv(Path(log_dir) / "agg_submit.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
