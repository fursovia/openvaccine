# OpenVaccine


```bash
TRAIN_DATA_PATH=./data/covid/train.jsonl \
    VALID_DATA_PATH=./data/covid/valid.jsonl \
    allennlp train ./configs/base.jsonnet \
    --serialization-dir ./logs/test_2 \
    --include-package openvaccine
```


```bash
allennlp predict logs/test/model.tar.gz data/test.json \
    --output-file preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --file-friendly-logging
```