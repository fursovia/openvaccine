#!/usr/bin/env bash

CONFIG_NAME=$1
TRAIN_PATH=${2:-"./data/train.jsonl"}
VALID_PATH=${3:-"./data/valid.jsonl"}
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}


TRAIN_DATA_PATH=${TRAIN_PATH} \
    VALID_DATA_PATH=${VALID_PATH} \
    allennlp train ./configs/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${EXP_NAME} \
    --include-package openvaccine
