#!/usr/bin/env bash

LOGDIR=$1
DATA_DIR=${2:-"./data/kfold"}
SUBMIT_NAME=$(basename ${LOGDIR})

for dir in ${DATA_DIR}/* ; do
    fold=$(basename $dir)
    VALID_PATH=${dir}/valid.jsonl
    PREDS_PATH=${LOGDIR}/${fold}/valid_preds.json

    allennlp predict ${LOGDIR}/${fold}/model.tar.gz \
        ${VALID_PATH} \
        --output-file ${PREDS_PATH} \
        --include-package openvaccine \
        --predictor covid_predictor \
        --use-dataset-reader \
        --cuda-device 0

    PYTHONPATH=. python openvaccine/commands/evaluate.py \
        ${PREDS_PATH} \
        ${VALID_PATH} \
        --out-path ${LOGDIR}/${fold}/metrics.json
done


PYTHONPATH=. python openvaccine/commands/aggregate_metrics.py ${LOGDIR}