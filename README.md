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

# Download wiki traffic data from Kaggle. Extract the archive
cd data/wiki && unzip wiki.zip
unzip key_1.csv.zip
unzip key_2.csv.zip
unzip train_1.csv.zip
unzip train_2.csv.zip
unzip sample_submission_1.csv.zip
unzip sample_submission_2.csv.zip
rm -rfv *.zip

mongod --bind_ip_all --dbpath data/mongodb --wiredTigerCacheSizeGB 10

# Scrape all page revisions from wiki
python scripts/get_wiki.py

# Download wiki dump
python scripts/download_wikidump.py

# With wiki dump, we get 668 files. Each file has on average 290M lines.
# If we use a single thread (no parallelization), it takes between 3-7 hours
# to go through each file. I've gone through 16 files and go 6800 articles.
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host dijkstra --n-jobs 39 --total 336 --split 0
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host localhost --n-jobs 39 --total 336 --split 1
python scripts/extract_graph.py --reindex

git clone --recurse-submodules -j8 https://github.com/RedisGraph/RedisGraph.git

python scripts/get_traffic.py -m dijkstra -b 0 -t 2 # cray
python scripts/get_traffic.py -m dijkstra -b 1 -t 2 # braun

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
