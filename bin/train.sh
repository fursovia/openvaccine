#!/usr/bin/env bash

CONFIG_NAME=$1
DATE=$(date +%T-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}


TRAIN_DATA_PATH=./data/train.jsonl \
    VALID_DATA_PATH=./data/valid.jsonl \
    allennlp train ./configs/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${EXP_NAME} \
    --include-package openvaccine
