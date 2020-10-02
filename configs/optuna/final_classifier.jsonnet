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

local bpps_dir = std.extVar('bpps_dir');
local sequence_emb_dim = std.parseInt(std.extVar('sequence_emb_dim'));
local structure_emb_dim = std.parseInt(std.extVar('structure_emb_dim'));
local predicted_loop_type_emb_dim = std.parseInt(std.extVar('predicted_loop_type_emb_dim'));

local bpps_aggegator = std.extVar('bpps_aggegator');
local bpps_dims = {
  "max_mean_sum_agg": 3,
  "max_sum_nb_agg": 3,
  "cnn_max_sum_nb_agg": 7,
  "null": 0,
};

local input_size = sequence_emb_dim + structure_emb_dim + predicted_loop_type_emb_dim + bpps_dims[bpps_aggegator];

local gru_hidden_size = std.parseInt(std.extVar('gru_hidden_size'));
local gru_num_layers = std.parseInt(std.extVar('gru_num_layers'));
local gru_dropout = std.parseJson(std.extVar('gru_dropout'));

local lstm_hidden_size = std.parseInt(std.extVar('lstm_hidden_size'));
local lstm_num_layers = std.parseInt(std.extVar('lstm_num_layers'));
local lstm_dropout = std.parseJson(std.extVar('lstm_dropout'));

//local augmented_lstm_hidden_size = std.parseInt(std.extVar('augmented_lstm_hidden_size'));
//local augmented_lstm_dropout = std.parseJson(std.extVar('augmented_lstm_dropout'));

local alternating_lstm_hidden_size = std.parseInt(std.extVar('alternating_lstm_hidden_size'));
local alternating_lstm_num_layers = std.parseInt(std.extVar('alternating_lstm_num_layers'));
local alternating_lstm_dropout = std.parseJson(std.extVar('alternating_lstm_dropout'));

local same_cnn_encoder_out_dim = std.parseInt(std.extVar('same_cnn_encoder_out_dim'));
local same_cnn_encoder_kernel_size = std.parseInt(std.extVar('same_cnn_encoder_kernel_size'));


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
//    "augmented_lstm": {
//      "type": "augmented_lstm",
//      "input_size": input_size,
//      "hidden_size": augmented_lstm_hidden_size,
//      "recurrent_dropout_probability": augmented_lstm_dropout,
//      "use_highway": true,
//    },
    "alternating_lstm": {
      "type": "alternating_lstm",
      "input_size": input_size,
      "hidden_size": alternating_lstm_hidden_size,
      "num_layers": alternating_lstm_num_layers,
      "recurrent_dropout_probability": alternating_lstm_dropout,
      "use_highway": true,
    },
    "same_cnn_encoder": {
      "type": "same_cnn_encoder",
      "input_dim": input_size,
      "out_dim": same_cnn_encoder_out_dim,
      "kernel_size": same_cnn_encoder_kernel_size,
      "bidirectional": true,
    },
};

local encoders = {
    "gru": base_encoders["gru"],
    "lstm": base_encoders["lstm"],
//    "augmented_lstm": base_encoders["augmented_lstm"],
    "alternating_lstm": base_encoders["alternating_lstm"],
    "same_cnn_encoder": base_encoders["same_cnn_encoder"],
    "stack_lstm_gru": {
      "type": "stack",
      "encoders": [
        base_encoders["lstm"],
        base_encoders["gru"],
      ]
    },
//    "stack_augmented_lstm_gru": {
//      "type": "stack",
//      "encoders": [
//        base_encoders["augmented_lstm"],
//        base_encoders["gru"],
//      ]
//    },
    "stack_alternating_lstm_gru": {
      "type": "stack",
      "encoders": [
        base_encoders["alternating_lstm"],
        base_encoders["gru"],
      ]
    },
    "stack_same_cnn_encoder_gru": {
      "type": "stack",
      "encoders": [
        base_encoders["same_cnn_encoder"],
        base_encoders["gru"],
      ]
    },
    "stack_same_cnn_encoder_lstm": {
      "type": "stack",
      "encoders": [
        base_encoders["same_cnn_encoder"],
        base_encoders["lstm"],
      ]
    },
//    "compose_same_cnn_encoder_stack_lstm_gru": {
//      "type": "compose",
//      "encoders": [
//        base_encoders["same_cnn_encoder"],
//        {
//          "type": "stack",
//          "encoders": [
//            base_encoders["same_cnn_encoder"],
//            base_encoders["lstm"],
//          ]
//        },
//    },
};

local encoder = std.extVar('encoder');


local structure_field_attention = std.extVar('structure_field_attention');
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

local predicted_loop_type_field_attention = std.extVar('predicted_loop_type_field_attention');
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


local masked_lm = std.extVar('masked_lm');
local lm_is_trainable = std.extVar('lm_is_trainable');
local LanguageModel(path='presets/lm.model.tar.gz') = {
  "type": "from_archive",
  "archive_file": path
};


local predicted_loop_type_field_attention = std.extVar('predicted_loop_type_field_attention');
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


local lm_dropout = std.parseJson(std.extVar('lm_dropout'));
local emb_dropout = std.parseJson(std.extVar('emb_dropout'));
local bpp_dropout = std.parseJson(std.extVar('bpp_dropout'));

{
  "dataset_reader": {
    "type": "covid_reader",
    "bpps_dir": if bpps_dir == 'null' then null else bpps_dir,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": VOCAB['vocabulary'],
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
      "type": "MCRMSE",
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