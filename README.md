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
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

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
python scripts/extract_graph.py --start 0 --end 270
python scripts/extract_graph.py --host dijkstra --start 270 --end 540
python scripts/extract_graph.py --host dijkstra --start 540 --end 668 --n_jobs 18
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
