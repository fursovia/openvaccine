local VOCAB = import 'common/vocab.jsonnet';

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
          "embedding_dim": 16,
          "trainable": true,
          "vocab_namespace": "sequence"
        }
      }
    },
    "structure_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 16,
          "trainable": true,
          "vocab_namespace": "structure"
        }
      }
    },
    "predicted_loop_type_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 16,
          "trainable": true,
          "vocab_namespace": "predicted_loop_type"
        }
      }
    },
    "seq2seq_encoder": {
      "type": "gated-cnn-encoder",
      "input_dim": 48,
      "layers": [ [[4, 48]], [[4, 48], [4, 48]], [[4, 48], [4, 48]] ],
      "dropout": 0.2,
    },
    "loss": {
      "type": "MCRMSE",
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
  "data_loader": {
    "batch_size": 1024,
    "shuffle": true,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0
  }
}