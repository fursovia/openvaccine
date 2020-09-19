#!/usr/bin/env bash

CONFIG_NAME=$1
TRAIN_FILENAME=${2:-"train"}
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}


TRAIN_DATA_PATH=./data/${TRAIN_FILENAME}.jsonl \
    VALID_DATA_PATH=./data/valid.jsonl \
    allennlp train ./configs/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${EXP_NAME} \
    --include-package openvaccine
