import json
from pathlib import Path

import typer
import pandas as pd

METRICS = ("MCRMSE", "best_validation_loss")


def main(log_dir: str, metrics_name: str = "kaggle_metrics.json"):

    metrics = []
    for path in Path(log_dir).iterdir():

        with open(str(path / metrics_name)) as f:
            curr_metrics = json.load(f)

        curr_metrics["fold"] = path.name
        metrics.append(curr_metrics)

    metrics = pd.DataFrame(metrics)

    for metric_name in METRICS:
        if metric_name in metrics.columns:
            for idx, row in metrics.iterrows():
                value = row[metric_name]
                typer.secho(f"({row.fold}) {metric_name} = {value:.3f}", fg="red")

            mean_value = metrics[metric_name].mean()
            typer.secho(f"{metric_name} = {mean_value:.3f}", fg="green")


if __name__ == "__main__":
    typer.run(main)
