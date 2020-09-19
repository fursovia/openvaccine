#!/usr/bin/env bash

LOGDIR=$1
VALID_PATH=${2:-"./data/valid.jsonl"}
SUBMIT_NAME=$(basename ${LOGDIR})


allennlp predict ${LOGDIR}/model.tar.gz \
    ${VALID_PATH} \
    --output-file ${LOGDIR}/valid_preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --use-dataset-reader \
    --cuda-device 0

PYTHONPATH=. python openvaccine/commands/evaluate.py \
    ${LOGDIR}/valid_preds.json \
    ./data/valid.jsonl
