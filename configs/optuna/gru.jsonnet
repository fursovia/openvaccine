local VOCAB = {
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
  }
};

local LOADER = {
  "data_loader": {
    "batch_size": 64,
    "shuffle": true,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  }
};

local sequence_emb_dim = std.parseInt(std.extVar("sequence_emb_dim"));
local structure_emb_dim = std.parseInt(std.extVar('structure_emb_dim'));
local predicted_loop_type_emb_dim = std.parseInt(std.extVar('predicted_loop_type_emb_dim'));

local hidden_size = std.parseInt(std.extVar('hidden_size'));
local num_layers = std.parseInt(std.extVar('num_layers'));
local gru_dropout = std.parseJson(std.extVar('gru_dropout'));
//local bidirectional_raw = std.extVar('bidirectional');
//local bidirectional = if bidirectional_raw == 'true' then true else false;
local bidirectional = true;

local variational_dropout = std.parseJson(std.extVar('variational_dropout'));
//local activation_raw = std.extVar('activation');
//local activation = if activation_raw == 'null' then null else std.parseJson(activation_raw);
local activation = 'relu';

local lr = std.parseJson(std.extVar('lr'));

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
          "embedding_dim": sequence_emb_dim,
          "trainable": true,
          "vocab_namespace": "sequence"
        }
      }
    },
    "structure_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": structure_emb_dim,
          "trainable": true,
          "vocab_namespace": "structure"
        }
      }
    },
    "predicted_loop_type_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": predicted_loop_type_emb_dim,
          "trainable": true,
          "vocab_namespace": "predicted_loop_type"
        }
      }
    },
    "seq2seq_encoder": {
      "type": "gru",
      "input_size": sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim,
      "hidden_size": hidden_size,
      "num_layers": num_layers,
      "dropout": gru_dropout,
      "bidirectional": bidirectional
    },
    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    },
    "variational_dropout": variational_dropout,
    "matrix_attention": {
      "type": "linear",
      "tensor_1_dim": if bidirectional then hidden_size * 2 else hidden_size,
      "tensor_2_dim": sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim,
      "combination": "x,y",
      "activation": activation
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
  "data_loader": LOADER['data_loader'],
  "trainer": {
    "num_epochs": 200,
    "patience": 15,
    "optimizer": {
      "type": "adam",
      "lr": lr
    },
    "cuda_device": 0
  }
}