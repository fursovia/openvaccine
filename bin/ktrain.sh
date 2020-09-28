#!/usr/bin/env bash

CONFIG_NAME=$1
DATA_DIR=${2:-"./data/kfold"}
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}


for dir in ${DATA_DIR}/* ; do
    fold=$(basename $dir)
    TRAIN_DATA_PATH=${dir}/train.jsonl \
        VALID_DATA_PATH=${dir}/valid.jsonl \
        allennlp train ./configs/${CONFIG_NAME}.jsonnet \
        --serialization-dir ./logs/${EXP_NAME}/${fold} \
        --include-package openvaccine
done
