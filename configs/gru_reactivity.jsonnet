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
    "type": "reactivity_classifier",
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
    "seq2seq_encoder": {
      "type": "gru",
      "input_size": 128,
      "hidden_size": 128,
      "num_layers": 2,
      "dropout": 0.3,
      "bidirectional": true
    }
  },
  "data_loader": {
    "batch_size": 256,
    "shuffle": true,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 200,
    "patience": 15,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": 0
  }
}