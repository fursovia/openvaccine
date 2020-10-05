#!/usr/bin/env bash

CONFIG_NAME=$1
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}


LM_TRAIN_DATA_PATH=./data/external/train.jsonl \
    LM_VALID_DATA_PATH=./data/external/valid.jsonl \
    allennlp train ./configs/lm/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${EXP_NAME} \
    --include-package openvaccine
