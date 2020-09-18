import pandas as pd


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