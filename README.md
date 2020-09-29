# OpenVaccine


```bash
CUDA_VISIBLE_DEVICES="1" bash bin/train.sh gru
```

```bash
CUDA_VISIBLE_DEVICES="1" bash bin/lm_train.sh bert_lm
```

```bash
CUDA_VISIBLE_DEVICES="1" bash bin/predict.sh LOGDIR
```

```bash
CUDA_VISIBLE_DEVICES="1" bash bin/evaluate.sh LOGDIR
```

```bash
git rev-parse --short HEAD
```


```bash
CUDA_VISIBLE_DEVICES="3" \
    TRAIN_DATA_PATH=./data/train.jsonl \
    VALID_DATA_PATH=./data/valid.jsonl \
    PYTHONPATH=. python openvaccine/commands/run_optuna.py \
        ./configs/optuna/gru.jsonnet \
        logs/optuna
```

## To-Do

* Calculate MCRMSE on the full dataset (not batches)
* Seq2seq for each sequence and a concatenation at the end
* Three (five) dense layers and the end (for each sequence)
* Optuna
* Pre-train LM
* Concat seq2seq output with embeddings
* Add attention mechanism for recurrent NNs
* Features using autoencoders
* Try BiMPM model
* Aggregate using GatedSum
* InputVariationalDropout, Attention, LayerNorm, MatrixAttention, ResidualWithLayerDropout, Maxout, ScalarMix
* Train on parsed data (reactivity)
* Estimate how external data helps 
(two commits: one with external data, another is without this data. Only reactivity!)
* Do a cross-validation (need to write some bash scripts) [https://www.kaggle.com/vbmokin/gru-lstm-mix-custom-loss-tuning-by-3d-visual]
* Find the perfect learning rate
* Why there are many reactivity observations?
* One final submit should be 130-oriented



## List of papers

* [Prediction of mRNA subcellular localization using deep recurrent neural networks](https://watermark.silverchair.com/btz337.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsEwggK9BgkqhkiG9w0BBwagggKuMIICqgIBADCCAqMGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMGqPSeLcY2Yi5luhHAgEQgIICdC8mFPy9EjTmY8QGolpAosWne9lqhWeheUIPLC5wt5ZaTj_FVegdjozlXfWTjoWayufwhlSBS9H6AAiITHeVWxVcNw3aZwJOFYhx9OlZBNHEoDxKRTdqkGWU6_iiletT1mLq5yi7EPTnuGKIC8aPo2mdISUoWzleIFLe4Y6QN6Z4VBtdREnngzkmDajdSaon6VEag-PKSH8R_Ggx9L7PYN57aZUUtyUAX8EvS7i1_L4zo253R1ePddsZnnJ8xykBjF5Yv1Mit9UbDXgXFWwNrQmWHejq_4sDFXg24ohUINi1yNlGqKZK0GHfPWIMLyR9mwfQZUw1hEzIKXRTnxEyweOgtX1c6RhobjpHkFcc2A6BM7ilcoFg2C7o_Op-4R6YhZUhn57Z2jmHfiVxWcgPk4bw5KOmO0W6fPgLaKgYxtcdl8kAztISt0ZeIYHB1sq2gfmbr8zNogFvQQKxClElLDu8firD3_9JP9-ranqeSIgj4SSbgBSZYE-UeDmAe5USWob4nb0XdYbE_uzw6eoL9dSw14_0oiVYVyLyDuDZrJ7ZdI7wkfcNZPzOzrgdad_upUOh7SRW4939XzIiCFzxDfuanEtIrKABRXbSUnuuzSltti81i0iTe6xgECcP_fYb34ShEbNZMZbvXtBMiLsf1HfTCYbD7VRNzvx0l6NC3sazyGDIUVUcqMDYzEVt-yvRoZ-J3lyZKHXrMap_o1w8M_OxIwi1WJ39g4tRPbCib5CxTJtv4gaOkdGymmDIohgxf2DxcqOClht-HRHMQsmuNFS6cswsCkv12TMkfQtZ2lcOhueTmdRFeKtTK8CvFzujRFy-UN4)
* [circDeep: deep learning approach for
circular RNA classification from other
long non-coding RNA](https://watermark.silverchair.com/btz537.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAr0wggK5BgkqhkiG9w0BBwagggKqMIICpgIBADCCAp8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMnkyj0NDZpP_NdO_wAgEQgIICcIleXKWdemFS5DzDYLNbJ4dwIWGoV4mh9bIq37ZtQ-lHt3IaRdIyayY6iKKJnjjvkFUQ37iqB4htlcTzaOs7npgwTDxg3Q51IPO2O3COPf5MdgXzx7ria-3CApBnJPI7hk4tBeLqfh6WCL2BQkWPKAbqNfVB5MvY4kyIEAK2_VwUnO9h5wT_kD3gQxNIiyX2f-oluFyjoajf8Fd4MSQj0kLcLPOUnJAT_DxaC4_ZypmCi-kJUjpbNtyil1HA-LRzZm8e3FO5d3acaYaDsYcJ92hPFi-uiWL3oSQ-8keTWWjT1VjO6zBS2v_yoXRIlbF6plfTFafcDou5iI2nZB672HAxcusLNZTD8rAXURnnGGKe2ow98qV4kGcXtaPmjQxcB84ilNQwHoIUp5ZFyuLoFCj2ovq8hwPVPGsaL4nbaPI9jQ_o6PB2Aoe9GCbk4g9MgH329GKf8aYuYUicRRtYtwQI6HAKFwvjcmgXkytfFu2Q4mX0ENDQGgVmpi_7FWKAFY9QdoHkj2S_IsGQ88URxFKPRLyAbbtEGWCOjT1xQtS4hhkIGbBqr57hs7sB9_yCoofS4re14KHxLhLCDGUa0CR8mu8rqnA4HVvypnM4XmU1t0Qm4sOencljSAMm2Rjmzl5SyyDI6TQvqSUIWPFvOIgdrRboQvZY6aXi8y6fZf_-ntCldsZwLOhXz0j4ez0AJb7IJkdTS7LclWGV088XiwBH3cOhZdNt5QAxXQ4llSqkvls1HALLczL7drjy3pfwGvZ5M10zV4kacdOYVsii5R7F1u1D9NYtYiMILXKj6qYBlcWTJcXLdFu6rgOpW3i1oA)
* [cDeepbind: A context sensitive deep learning
model of RNA-protein binding](https://www.biorxiv.org/content/10.1101/345140v1.full.pdf)
* [Uncovering the Folding Landscape of RNA
Secondary Structure with Deep Graph Embeddings](https://arxiv.org/pdf/2006.06885.pdf)
* [A Novel Hybrid CNN-SVR for CRISPR/Cas9 Guide RNA Activity Prediction](https://www.frontiersin.org/articles/10.3389/fgene.2019.01303/full)

## List of datasets

