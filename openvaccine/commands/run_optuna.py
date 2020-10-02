from functools import partial
import os

import typer
import optuna


BASE_PARAMS = {
    "gru": {
        "gru_hidden_size": 128,
        "gru_num_layers": 2,
        "gru_dropout": 0.4
    },
    "lstm": {
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.4
    },
    # "augmented_lstm": {
    #     "augmented_lstm_hidden_size": 128,
    #     "augmented_lstm_dropout": 0.4
    # },
    "alternating_lstm": {
        "alternating_lstm_hidden_size": 128,
        "alternating_lstm_num_layers": 128,
        "alternating_lstm_dropout": 0.4
    },
    "same_cnn_encoder": {
        "same_cnn_encoder_out_dim": 128,
        "same_cnn_encoder_kernel_size": 5,
    },

}


NAMES_TO_ENCODERS = {
    "gru": ["gru"],
    "lstm": ["lstm"],
    # "augmented_lstm": ["augmented_lstm"],
    "alternating_lstm": ["alternating_lstm"],
    "same_cnn_encoder": ["same_cnn_encoder"],
    "stack_lstm_gru": ["lstm", "gru"],
    # "stack_augmented_lstm_gru": ["augmented_lstm", "gru"],
    "stack_alternating_lstm_gru": ["alternating_lstm", "gru"],
    "stack_same_cnn_encoder_gru": ["same_cnn_encoder", "gru"],
    "stack_same_cnn_encoder_lstm": ["same_cnn_encoder", "lstm"],
}


def set_trial(trial: optuna.Trial):
    bpps_dir = trial.suggest_categorical("bpps_dir", ["null", "data/raw_data/bpps"])
    if bpps_dir != "null":
        # "max_sum_nb_agg", "cnn_max_sum_nb_agg"
        trial.suggest_categorical("bpps_aggegator", ["null", "max_mean_sum_agg", ])
        trial.suggest_float("bpp_dropout", 0.0, 0.7)
    else:
        os.environ["bpps_aggegator"] = "null"
        os.environ["bpp_dropout"] = "0.0"

    trial.suggest_categorical("sequence_emb_dim", [16, 32, 64, 128])
    trial.suggest_categorical("structure_emb_dim", [16, 32, 64, 128])
    trial.suggest_categorical("predicted_loop_type_emb_dim", [16, 32, 64, 128])
    trial.suggest_float("emb_dropout", 0.0, 0.7)

    encoder = trial.suggest_categorical(
        "encoder",
        [
            "gru",
            "lstm",
            # "augmented_lstm",
            "alternating_lstm",
            "same_cnn_encoder",
            "stack_lstm_gru",
            # "stack_augmented_lstm_gru",
            "stack_alternating_lstm_gru",
            "stack_same_cnn_encoder_gru",
            "stack_same_cnn_encoder_lstm"
        ]
    )

    if encoder == "gru":
        trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("gru_num_layers", [1, 2, 3])
        trial.suggest_float("gru_dropout", 0.0, 0.7)

    elif encoder == "lstm":
        trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("lstm_num_layers", [1, 2, 3])
        trial.suggest_float("lstm_dropout", 0.0, 0.7)

    # elif encoder == "augmented_lstm":
    #     trial.suggest_categorical("augmented_lstm_hidden_size", [16, 32, 64, 128])
    #     trial.suggest_float("augmented_lstm_dropout", 0.0, 0.7)

    elif encoder == "alternating_lstm":
        trial.suggest_categorical("alternating_lstm_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("alternating_lstm_num_layers", [1, 2, 3])
        trial.suggest_float("alternating_lstm_dropout", 0.0, 0.7)

    elif encoder == "same_cnn_encoder":
        trial.suggest_categorical("same_cnn_encoder_out_dim", [16, 32, 64, 128])
        trial.suggest_categorical("same_cnn_encoder_kernel_size", [2, 3, 4, 5, 6, 7])

    elif encoder == "stack_lstm_gru":
        trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("lstm_num_layers", [1, 2, 3])
        trial.suggest_float("lstm_dropout", 0.0, 0.7)

        trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("gru_num_layers", [1, 2, 3])
        trial.suggest_float("gru_dropout", 0.0, 0.7)

    # elif encoder == "stack_augmented_lstm_gru":
    #     trial.suggest_categorical("augmented_lstm_hidden_size", [16, 32, 64, 128])
    #     trial.suggest_float("lstm_dropout", 0.0, 0.7)
    #
    #     trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128])
    #     trial.suggest_categorical("gru_num_layers", [1, 2, 3])
    #     trial.suggest_float("gru_dropout", 0.0, 0.7)

    elif encoder == "stack_alternating_lstm_gru":
        trial.suggest_categorical("alternating_lstm_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("alternating_lstm_num_layers", [1, 2, 3])
        trial.suggest_float("alternating_lstm_dropout", 0.0, 0.7)

        trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("gru_num_layers", [1, 2, 3])
        trial.suggest_float("gru_dropout", 0.0, 0.7)

    elif encoder == "stack_same_cnn_encoder_gru":
        trial.suggest_categorical("same_cnn_encoder_out_dim", [16, 32, 64, 128])
        trial.suggest_categorical("same_cnn_encoder_kernel_size", [2, 3, 4, 5, 6, 7])

        trial.suggest_categorical("gru_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("gru_num_layers", [1, 2, 3])
        trial.suggest_float("gru_dropout", 0.0, 0.7)

    elif encoder == "stack_same_cnn_encoder_lstm":
        trial.suggest_categorical("same_cnn_encoder_out_dim", [16, 32, 64, 128])
        trial.suggest_categorical("same_cnn_encoder_kernel_size", [2, 3, 4, 5, 6, 7])

        trial.suggest_categorical("lstm_hidden_size", [16, 32, 64, 128])
        trial.suggest_categorical("lstm_num_layers", [1, 2, 3])
        trial.suggest_float("lstm_dropout", 0.0, 0.7)

    for name, params in BASE_PARAMS.items():
        if name not in NAMES_TO_ENCODERS[encoder]:
            for key, val in params.items():
                os.environ[key] = str(val)

    trial.suggest_categorical("structure_field_attention", ["linear", "bilinear", "null"])
    trial.suggest_categorical("predicted_loop_type_field_attention", ["linear", "bilinear", "null"])

    masked_lm = trial.suggest_categorical("masked_lm", ["null", "presets/lm.model.tar.gz"])
    if masked_lm != "null":
        trial.suggest_categorical("lm_is_trainable", ["true", "false"])
    else:
        os.environ["lm_is_trainable"] = "false"

    trial.suggest_float("lm_dropout", 0.0, 0.7)


def openvaccine_objective(
        trial: optuna.Trial,
        config_path: str,
        serialization_dir: str
) -> float:
    set_trial(trial)

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
        timeout: int = 60 * 60 * 24,
        study_name: str = "optuna_openvaccine"
):
    study = optuna.create_study(
        storage="sqlite:///result/final_classifier.db",
        sampler=optuna.samplers.TPESampler(seed=245),
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
