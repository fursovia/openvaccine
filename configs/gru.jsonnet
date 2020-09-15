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
    "type": "covid_classifier",
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
      "dropout": 0.3,
      "bidirectional": true
    },
    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    }
  },
//  "distributed": {
//    "master_address": "127.0.0.1",
//    "master_port": 29502,
//    "num_nodes": 1,
//    "cuda_devices": [
//      0,
//      1
//    ]
//  },
  "data_loader": LOADER['data_loader'],
  "trainer": {
    "num_epochs": 200,
    "patience": 15,
    "cuda_device": 0
  }
}