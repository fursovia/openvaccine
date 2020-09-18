local LOADER = import 'common/loader.jsonnet';

{
  "dataset_reader": {
    "type": "covid_reader",
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
    "vocabulary": {
    "type": "extend",
    "directory": "presets/vocabulary"
  },
  "model": {
    "type": "covid_classifier",
    "masked_lm": {
      "type": "from_archive",
      "archive_file": "presets/lm.model.tar.gz"
    },
    "lm_is_trainable": false,
    "lm_matrix_attention": {
      "type": "linear",
      "tensor_1_dim": 256,
      "tensor_2_dim": 128,
      "combination": "x,y",
      "activation": null
    },
    "sequence_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 64,
          "trainable": true,
          "vocab_namespace": "sequence"
        }
      }
    },
    "structure_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 64,
          "trainable": true,
          "vocab_namespace": "structure"
        }
      }
    },
    "predicted_loop_type_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 64,
          "trainable": true,
          "vocab_namespace": "predicted_loop_type"
        }
      }
    },
    "seq2seq_encoder": {
      "type": "gru",
      "input_size": 192,
      "hidden_size": 128,
      "num_layers": 2,
      "dropout": 0.4,
      "bidirectional": true
    },
    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    },
    "variational_dropout": 0.1
  },
  "data_loader": LOADER['data_loader'],
  "trainer": {
    "num_epochs": 200,
    "patience": 15,
    "optimizer": {
      "type": "adam",
      "lr": 0.0005
    },
    "cuda_device": 0
  }
}