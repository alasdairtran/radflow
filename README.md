# nos

Network of sequences

## Getting Started

```sh
# Install package and dependencies
python setup.py develop
```

## Training

```sh
# Naive baselines (no training is needed)
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/1_naive_previous_day/config.yaml
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/2_naive_seasonal/config.yaml

# LSTM baseline
CUDA_VISIBLE_DEVICES=0 nos train expt/7_lstm/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/7_lstm/config.yaml -m expt/7_lstm/serialization/best.th

# Network baseline
CUDA_VISIBLE_DEVICES=0 nos train expt/8_lstm_with_neighbours/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/8_lstm_with_neighbours/config.yaml \
    -m expt/8_lstm_with_neighbours/serialization/best.th \
    --overrides '{"dataset_reader": {"evaluate_mode": true}, "model": {"evaluate_mode": true}}'
```
