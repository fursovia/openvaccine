{
  "dataset_reader": {
    "type": "covid_reader",
    "lazy": false
  },
  "train_data_path": std.extVar("LM_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("LM_VALID_DATA_PATH"),
  "vocabulary": {
    "tokens_to_add": {
      "sequence": [
        "@@MASK@@",
        "<START>",
        "<END>"
      ],
    "structure": [
        "<START>",
        "<END>"
      ]
    },
  },
  "model": {
    "type": "masked_lm",
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
      "type": "pytorch_transformer",
      "input_dim": 128,
      "num_layers": 3,
      "num_attention_heads": 4,
      "positional_encoding": "embedding"
    },
    "tokens_masker": {
      "type": "tokens_masker",
      "mask_probability": 0.3,
      "replace_probability": 0.1
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
    "num_epochs": 100,
    "patience": 3,
    "cuda_device": 0
  }
}