local VOCAB = {
    "type": "extend",
    "directory": "presets/vocabulary"
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

local sequence_emb_dim = 64;
local structure_emb_dim = 64;
local predicted_loop_type_emb_dim = 64;


local bpps_dir = "null";
//local bpps_dir = "data/raw_data/bpps";
local bpp_dropout = 0.1;
// should try to change
local bpps_aggegator = "null";
//local bpps_aggegator = "max_mean_nb_agg";
local bpps_dims = {
  "max_mean_sum_agg": 3,
  "max_mean_agg": 2,
  "max_mean_nb_agg": 3,
  "null": 0,
};

local input_size = sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim + bpps_dims[bpps_aggegator];

local gru_hidden_size = 128;
local gru_num_layers = 3;
local gru_dropout = 0.35;

// this is for stack_lstm_gru
local lstm_hidden_size = 16;
// 3 is ok too
local lstm_num_layers = 1;
local lstm_dropout = 0.25;


// this is for lstm
//local lstm_hidden_size = 128;
//local lstm_num_layers = 3;
//local lstm_dropout = 0.23;

// this is for stack_alternating_lstm_gru
local alternating_lstm_hidden_size = 32;
local alternating_lstm_num_layers = 1;
local alternating_lstm_dropout = 0.60;



local base_encoders = {
    "gru": {
      "type": "gru",
      "input_size": input_size,
      "hidden_size": gru_hidden_size,
      "num_layers": gru_num_layers,
      "dropout": gru_dropout,
      "bidirectional": true
    },
    "lstm": {
      "type": "lstm",
      "input_size": input_size,
      "hidden_size": lstm_hidden_size,
      "num_layers": lstm_num_layers,
      "dropout": lstm_dropout,
      "bidirectional": true
    },
    "alternating_lstm": {
      "type": "alternating_lstm",
      "input_size": input_size,
      "hidden_size": alternating_lstm_hidden_size,
      "num_layers": alternating_lstm_num_layers,
      "recurrent_dropout_probability": alternating_lstm_dropout,
      "use_highway": true,
    },
};

local encoders = {
    "gru": base_encoders["gru"],
    "lstm": base_encoders["lstm"],
    "alternating_lstm": base_encoders["alternating_lstm"],
    "stack_lstm_gru": {
      "type": "stack",
      "encoders": [
        base_encoders["lstm"],
        base_encoders["gru"],
      ]
    },
    "stack_alternating_lstm_gru": {
      "type": "stack",
      "encoders": [
        base_encoders["alternating_lstm"],
        base_encoders["gru"],
      ]
    },
};

local encoder = "stack_lstm_gru";
//local encoder = "lstm";
//local encoder = "stack_alternating_lstm_gru";


local structure_field_attention = "null";
local structure_field_attentions = {
  "linear": {
      "type": "linear",
      "tensor_1_dim": sequence_emb_dim,
      "tensor_2_dim": structure_emb_dim,
      "combination": "x,y",
      "activation": "relu"
    },
  "bilinear": {
      "type": "bilinear",
      "matrix_1_dim": sequence_emb_dim,
      "matrix_2_dim": structure_emb_dim,
      "activation": "relu"
    },
  "null": null
};

local predicted_loop_type_field_attention = "null";
local predicted_loop_type_field_attentions = {
  "linear": {
      "type": "linear",
      "tensor_1_dim": sequence_emb_dim,
      "tensor_2_dim": predicted_loop_type_emb_dim,
      "combination": "x,y",
      "activation": "relu"
    },
  "bilinear": {
      "type": "bilinear",
      "matrix_1_dim": sequence_emb_dim,
      "matrix_2_dim": predicted_loop_type_emb_dim,
      "activation": "relu"
    },
  "null": null
};


//local masked_lm = "presets/lm.model.tar.gz";
local masked_lm = "presets/gru_lm.model.tar.gz";

local lm_is_trainable = "false";
local LanguageModel(path='presets/lm.model.tar.gz') = {
  "type": "from_archive",
  "archive_file": path
};


local predicted_loop_type_field_attention = "null";
local predicted_loop_type_field_attentions = {
  "linear": {
      "type": "linear",
      "tensor_1_dim": sequence_emb_dim,
      "tensor_2_dim": predicted_loop_type_emb_dim,
      "combination": "x,y",
      "activation": "relu"
    },
  "bilinear": {
      "type": "bilinear",
      "matrix_1_dim": sequence_emb_dim,
      "matrix_2_dim": predicted_loop_type_emb_dim,
      "activation": "relu"
    },
  "null": null
};


local lm_dropout = 0.55;
local emb_dropout = 0.17;

{
  "dataset_reader": {
    "type": "covid_reader",
    "bpps_dir": if bpps_dir == 'null' then null else bpps_dir,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": VOCAB,
  "model": {
    "type": "final_classifier",
    // embedders
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
    // encoder
    "seq2seq_encoder": encoders[encoder],

    "loss": {
      "type": "MSE",
      "calculate_on_scored": true
    },
    "structure_field_attention": structure_field_attentions[structure_field_attention],
    "predicted_loop_type_field_attention": predicted_loop_type_field_attentions[predicted_loop_type_field_attention],

    "masked_lm": if masked_lm == 'null' then null else LanguageModel(masked_lm),
    "lm_is_trainable": if lm_is_trainable == 'true' then true else false,
    "lm_matrix_attention": null,
    "lm_dropout": lm_dropout,
    "emb_dropout": emb_dropout,
    "bpps_aggegator": if bpps_aggegator == 'null' then null else bpps_aggegator,
    "bpp_dropout": bpp_dropout,
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
    "patience": 10,
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "factor": 0.5,
//      "mode": "min",
//      "patience": 2
//    },
    "optimizer": {
      "type": "adam",
      "lr": 0.0015
    },
    "cuda_device": 0
  }
}