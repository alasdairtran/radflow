# nos

Network of sequences

## Getting Started

```sh
conda env create -f conda.yaml
conda activate nos
python -m ipykernel install --user --name nos --display-name "nos"
cd lib/apex
git submodule init && git submodule update .
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../.. && python setup.py develop
pip install -U torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install -U torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install -U torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install -U torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install -U torch-geometric
git clone --recurse-submodules -j8 https://github.com/RedisGraph/RedisGraph.git

# Start an empty mongodb database
mongod --bind_ip_all --dbpath data/mongodb --wiredTigerCacheSizeGB 10

# Download wiki dump
python scripts/download_wikidump.py

# With wiki dump, we get 668 files. Each file has on average 290M lines.
# If we use a single thread (no parallelization), it takes between 3-7 hours
# to go through each file. The following scripts construct a mongo database
# for the entire wiki graph
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host dijkstra --n-jobs 39 --total 336 --split 0 # cray
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host localhost --n-jobs 39 --total 336 --split 1 # dijkstra
python scripts/extract_graph.py --reindex

# Get page view counts directly from wiki API
# Also create a redis database to cache a map from title to node ID
python scripts/get_traffic.py -m localhost -b 0 -t 3 # dijkstra
python scripts/get_traffic.py -m dijkstra -b 1 -t 3 # cray
python scripts/get_traffic.py -m dijkstra -b 2 -t 3 # braun

# Extract static subgraphs for our first round of experiment
python scripts/extract_static.py -r cray # dijsktra
```

## Training

```sh
# Naive baselines (no training is needed)
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/1_naive_previous_day/config.yaml
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/2_naive_seasonal/config.yaml
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/3_naive_rolling/config.yaml
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/4_naive_seasonal_diff/config.yaml

# LSTM baseline
CUDA_VISIBLE_DEVICES=0 nos train expt/7_lstm/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/7_lstm/config.yaml -m expt/7_lstm/serialization/best.th

# Network baseline
CUDA_VISIBLE_DEVICES=0 nos train expt/8_agg_attention/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/8_agg_attention/config.yaml -m expt/8_agg_attention/serialization/best.th

CUDA_VISIBLE_DEVICES=0 nos train expt/9_agg_sum/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/9_agg_sum/config.yaml -m expt/9_agg_sum/serialization/best.th

CUDA_VISIBLE_DEVICES=0 nos train expt/10_gcn/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/10_gcn/config.yaml -m expt/10_gcn/serialization/best.th

CUDA_VISIBLE_DEVICES=0 nos train expt/11_no_agg/config.yaml -f
CUDA_VISIBLE_DEVICES=0 nos evaluate expt/11_no_agg/config.yaml -m expt/11_no_agg/serialization/best.th
```
