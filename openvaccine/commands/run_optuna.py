from functools import partial

import typer
import optuna


def set_gru_trial(trial: optuna.Trial):
    trial.suggest_categorical("sequence_emb_dim", [4, 8, 16, 32, 64, 128])
    trial.suggest_categorical("structure_emb_dim", [4, 8, 16, 32, 64, 128])
    trial.suggest_categorical("predicted_loop_type_emb_dim", [4, 8, 16, 32, 64, 128])

    trial.suggest_categorical("hidden_size", [4, 8, 16, 32, 64, 128, 256])
    trial.suggest_categorical("num_layers", [1, 2, 3, 4])
    trial.suggest_float("gru_dropout", 0.0, 0.5)
    # trial.suggest_categorical("bidirectional", ["true", "false"])

    trial.suggest_float("variational_dropout", 0.0, 0.5)
    # trial.suggest_categorical("activation", ["relu", "null"])
    trial.suggest_float("lr", 0.0001, 0.1, log=True)


CONFIGS = {
    "gru": set_gru_trial
}


def openvaccine_objective(
        trial: optuna.Trial,
        config_path: str,
        serialization_dir: str
) -> float:
    config_name = config_path.split('/')[-1].split('.')[0]
    CONFIGS[config_name](trial)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,
        config_file=config_path,
        serialization_dir=f"{serialization_dir}/{trial.number}",
        metrics="best_validation_loss",
        include_package="openvaccine",
    )
    return executor.run()


def main(
        config_path: str,
        serialization_dir: str,
        num_trials: int = 500,
        n_jobs: int = 1,
        timeout: int = 60 * 60 * 10,
        study_name: str = "optuna_openvaccine"
):
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )

    objective = partial(openvaccine_objective, config_path=config_path, serialization_dir=serialization_dir)
    study.optimize(
        objective,
        n_jobs=n_jobs,
        n_trials=num_trials,
        timeout=timeout,
    )


if __name__ == "__main__":
    typer.run(main)
