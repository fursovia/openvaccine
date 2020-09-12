{
  "dataset_reader": {
    "type": "covid_reader",
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": {
    "tokens_to_add": {
      "sequence": [
        "<START>",
        "<END>"
      ],
      "structure": [
        "<START>",
        "<END>"
      ],
      "predicted_loop_type": [
        "<START>",
        "<END>"
      ]
    },
  },
  "model": {
    "type": "covid_classifier",
    "sequence_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 8,
          "trainable": true,
          "vocab_namespace": "sequence"
        }
      }
    },
    "structure_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 8,
          "trainable": true,
          "vocab_namespace": "structure"
        }
      }
    },
    "predicted_loop_type_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 8,
          "trainable": true,
          "vocab_namespace": "predicted_loop_type"
        }
      }
    },
    "seq2seq_encoder": {
      "type": "pytorch_transformer",
      "input_dim": 24,
      "num_layers": 4,
      "num_attention_heads": 2,
      "feedforward_hidden_dim": 32,
      "positional_embedding_size": 512,
      "positional_encoding": "embedding",
      "dropout_prob": 0.1,
      "activation": "relu"
    },

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
    "batch_size": 32,
    "shuffle": false,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": false
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 2,
    "cuda_device": -1
  }
}
