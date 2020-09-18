local VOCAB = import 'common/vocab.jsonnet';
local LOADER = import 'common/loader.jsonnet';

{
  "dataset_reader": {
    "type": "covid_reader",
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": VOCAB['vocabulary'],
  "model": {
    "type": "transfer_classifier",
    "masked_lm": {
      "type": "from_archive",
      "archive_file": "presets/lm.model.tar.gz"
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
    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    },
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