#!/usr/bin/env bash

LOGDIR=$1
SUBMIT_NAME=$(basename ${LOGDIR})


PYTHONPATH=. python openvaccine/commands/aggregate_csv_preds.py ${LOGDIR}
PYTHONPATH=. python openvaccine/commands/submit.py \
    ${LOGDIR}/test_preds.json \
    ${LOGDIR}/agg_${SUBMIT_NAME}_submit.csv
