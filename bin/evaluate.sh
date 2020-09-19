#!/usr/bin/env bash

LOGDIR=$1
SUBMIT_NAME=$(basename ${LOGDIR})

allennlp predict ${LOGDIR}/model.tar.gz \
    ./data/valid.jsonl \
    --output-file ${LOGDIR}/valid_preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --use-dataset-reader \
    --cuda-device 0

PYTHONPATH=. python openvaccine/commands/evaluate.py \
    ${LOGDIR}/valid_preds.json \
    ./data/valid.jsonl
