local VOCAB = import 'common/vocab.jsonnet';
local LOADER = import 'common/loader.jsonnet';

{
  "dataset_reader": {
    "type": "covid_reader",
    "bpps_dir": "data/raw_data/bpps",
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
      "type": "stacked_bidirectional_lstm",
      "input_size": 195,
      "hidden_size": 128,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.1,
      "layer_dropout_probability": 0.1,
      "use_highway": true
    },
    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    },
    "bpps_aggegator": "max_mean_sum_agg",
    "variational_dropout": 0.0,
    "matrix_attention": {
      "type": "linear",
      "tensor_1_dim": 256,
      "tensor_2_dim": 195,
      "combination": "x,y",
      "activation": null
    },
    "regularizer": {
      "regexes": [
        [".*", {
          "type": "l2",
          "alpha": 1e-07
        }]
      ]
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
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "factor": 0.5,
//      "mode": "min",
//      "patience": 2
//    },
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": 0
  }
}