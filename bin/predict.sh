#!/usr/bin/env bash

LOGDIR=$1
SUBMIT_NAME=$(basename ${LOGDIR})


allennlp predict ${LOGDIR}/model.tar.gz \
    ./data/test.json \
    --output-file ${LOGDIR}/test_preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --cuda-device 0
    # --batch-size 64


PYTHONPATH=. python openvaccine/commands/submit.py \
    ${LOGDIR}/test_preds.json \
    ${LOGDIR}/${SUBMIT_NAME}_submit.csv
