#!/usr/bin/env bash

LOGDIR=$1
SUBMIT_NAME=$(basename ${LOGDIR})


allennlp predict ${LOGDIR}/model.tar.gz \
    ./data/test.json \
    --output-file ${LOGDIR}/test_preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --use-dataset-reader \
    --cuda-device 0

bash bin/evaluate.sh ${LOGDIR}
cat ${LOGDIR}/metrics.json | grep best_validation_loss

PYTHONPATH=. python openvaccine/commands/submit.py \
    ${LOGDIR}/test_preds.json \
    ${LOGDIR}/${SUBMIT_NAME}_submit.csv

git rev-parse --short HEAD
