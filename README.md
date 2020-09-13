# OpenVaccine


```bash
CUDA_VISIBLE_DEVICES="1" bash bin/train.sh gru
```

```bash
TRAIN_DATA_PATH=./data/train.jsonl \
    VALID_DATA_PATH=./data/valid.jsonl \
    allennlp train ./configs/transformer.jsonnet \
    --serialization-dir ./logs/test \
    --include-package openvaccine
```


```bash
allennlp predict logs/test/model.tar.gz data/test.json \
    --output-file logs/test/test_preds.json \
    --include-package openvaccine \
    --predictor covid_predictor \
    --file-friendly-logging
```