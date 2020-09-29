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

local bpps_aggegator_raw = std.extVar('bpps_aggegator');
local bpps_aggegator = if bpps_aggegator_raw == 'null' then null else bpps_aggegator_raw;
local enable_loss_weights = std.extVar('enable_loss_weights');

local sequence_emb_dim = 128;
local structure_emb_dim = 64;
local predicted_loop_type_emb_dim = 32;

local hidden_size = 256;
local num_layers = 3;
local gru_dropout = 0.35;
//local bidirectional_raw = std.extVar('bidirectional');
//local bidirectional = if bidirectional_raw == 'true' then true else false;
local bidirectional = true;

local variational_dropout = 0.4;
local activation_raw = std.extVar('activation');
local activation = if activation_raw == 'null' then null else activation_raw;

local lr = 0.0015;

local aggregator_dims = {
  "max_mean_sum_agg": 3,
  "max_sum_nb_agg": 3,
  "cnn_max_sum_nb_agg": 7,
  "null": 0,
};

local input_dim = sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim + aggregator_dims[bpps_aggegator_raw];


local attentions = {
  "linear": {
      "type": "linear",
      "tensor_1_dim": if bidirectional then hidden_size * 2 + 3 else hidden_size,
      "tensor_2_dim": sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim,
      "combination": "x,y",
      "activation": activation
    },
  "bilinear": {
      "type": "bilinear",
      "matrix_1_dim": if bidirectional then hidden_size * 2 else hidden_size,
      "matrix_2_dim": input_dim,
      "activation": activation
    },
  "null": null
};


local reg_type = std.extVar('reg_type');
local alpha = std.parseJson(std.extVar('alpha'));


local regularizers = {
  "l1": {
      "regexes": [
        [".*", {
          "type": "l1",
          "alpha": alpha
        }]
      ]
    },
  "l2": {
      "regexes": [
        [".*", {
          "type": "l2",
          "alpha": alpha
        }]
      ]
    },
  "null": null
};


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
      "input_size": input_dim,
      "hidden_size": hidden_size,
      "num_layers": num_layers,
      "dropout": gru_dropout,
      "bidirectional": bidirectional
    },
    "loss": {
      "type": "MCRMSE",
      "calculate_on_scored": true
    },
    "bpps_aggegator": bpps_aggegator,
    "enable_loss_weights": if enable_loss_weights == 'true' then true else false,
    "variational_dropout": variational_dropout,
    "matrix_attention": attentions[activation_raw],
    "regularizer": regularizers[reg_type]
  },
  "data_loader": LOADER['data_loader'],
  "trainer": {
    "num_epochs": 2,
    "patience": 1,
    "optimizer": {
      "type": "adam",
      "lr": lr
    },
    "cuda_device": -1
  }
}