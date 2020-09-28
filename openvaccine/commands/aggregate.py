import json
from pathlib import Path

import typer
import pandas as pd


def main(log_dir: str):

    metrics = []
    for path in Path(log_dir).iterdir():

        with open(str(path / "metrics.json")) as f:
            curr_metrics = json.load(f)

        curr_metrics["fold"] = path.name
        metrics.append(curr_metrics)

    metrics = pd.DataFrame(metrics)
    for idx, row in metrics.iterrows():
        typer.secho(f"({row.fold}) MCRMSE = {row.MCRMSE:.3f}", fg="red")

    mcrmse = metrics["MCRMSE"].mean()
    typer.secho(f"MCRMSE = {mcrmse:.3f}", fg="green")


if __name__ == "__main__":
    typer.run(main)
