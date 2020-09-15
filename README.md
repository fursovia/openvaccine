# OpenVaccine

TO-DO:

* Calculate MCRMSE on the full dataset (not batches)
* Seq2seq for each sequence and a concatenation at the end


```bash
CUDA_VISIBLE_DEVICES="1" bash bin/train.sh gru
```

```bash
CUDA_VISIBLE_DEVICES="1" bash bin/predict.sh LOGDIR
```

```bash
git rev-parse --short HEAD
```