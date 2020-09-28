#!/usr/bin/env bash

LOGDIR=$1
SUBMIT_NAME=$(basename ${LOGDIR})


for dir in ${LOGDIR}/* ; do
    fold=$(basename $dir)
    PREDS_PATH=${LOGDIR}/${fold}/test_preds.json

    allennlp predict ${LOGDIR}/${fold}/model.tar.gz \
        ./data/test.json \
        --output-file ${PREDS_PATH} \
        --include-package openvaccine \
        --predictor covid_predictor \
        --use-dataset-reader \
        --cuda-device 0
done

bash bin/kevaluate.sh ${LOGDIR}
PYTHONPATH=. python openvaccine/commands/aggregate_metrics.py ${LOGDIR} --metrics-name "metrics.json"

PYTHONPATH=. python openvaccine/commands/aggregate_preds.py ${LOGDIR}
PYTHONPATH=. python openvaccine/commands/submit.py \
    ${LOGDIR}/test_preds.json \
    ${LOGDIR}/${SUBMIT_NAME}_submit.csv

git rev-parse --short HEAD
