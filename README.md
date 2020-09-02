# nos

Network of sequences

## Getting Started

```sh
conda env create -f conda.yaml
conda activate nos
python -m ipykernel install --user --name nos --display-name "nos"
python setup.py develop

# Install apex
git submodule init lib/apex && git submodule update --init lib/apex
cd lib/apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../..

# Install PyTorch Geometric
pip install -U torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -U torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -U torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -U torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install -U torch-geometric

# Install redis from source
cd lib
wget http://download.redis.io/releases/redis-6.0.6.tar.gz
tar xvzf redis-6.0.6.tar.gz
rm -rf redis-6.0.6.tar.gz
cd redis-6.0.6
make
make test
cd ../..

# Install RedisGraph
cd lib/RedisGraph && git submodule update --init --recursive .
git clone --recurse-submodules -j8 https://github.com/RedisGraph/RedisGraph.git
sudo apt install build-essential cmake m4 automake peg libtool autoconf
make
cd ../..

# Start an empty mongodb database
mongod --bind_ip_all --dbpath data/mongodb --wiredTigerCacheSizeGB 10

# Start redis server
lib/redis-6.0.6/src/redis-server \
    --protected-mode no \
    --loadmodule lib/RedisGraph/src/redisgraph.so \
    --overcommit-memory 1

# Download wiki dump. This takes about three days.
python scripts/download_wikidump.py

# With wiki dump, we get 668 files. Each file has on average 290M lines.
# If we use a single thread (no parallelization), it takes between 3-7 hours
# to go through each file. The following scripts construct a mongo database
# for the entire wiki graph. This takes about 40 hours.
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host dijkstra --n-jobs 24 --total 232 --split 0 # braun
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host dijkstra --n-jobs 20 --total 232 --split 1 # cray
python scripts/extract_graph.py --dump /data4/u4921817/nos/data/wikidump --host dijkstra --n-jobs 20 --total 232 --split 2 # cray

# Remove duplicate titles. Generate a cache title2pageid.pkl that maps
# the title to the original page id. We also reindex the page IDs, taking 3h.
# We end up with 17,380,550 unqiue IDs/titles.
python scripts/extract_graph.py --reindex

# Get page view counts directly from wiki API. Takes around 3 days.
python scripts/get_traffic.py -m localhost -b 0 -t 3 # dijkstra
python scripts/get_traffic.py -m dijkstra -b 1 -t 3 # cray
python scripts/get_traffic.py -m dijkstra -b 2 -t 3 # braun

# Store wiki graph in hdf5
python scripts/extract_wiki_subgraph.py

docker build -t alasdairtran/radflow .
docker push alasdairtran/radflow
```

## Training

```sh
# Some experiments don't utilize the whole GPU, so we can run many parallel
# experiments on the same GPU.
# When using MPS it is recommended to use EXCLUSIVE_PROCESS mode to ensure that
# only a single MPS server is using the GPU, which provides additional insurance that the
# MPS server is the single point of arbitration between all CUDA processes for that GPU.
# Setting this does not persist across reboot
sudo nvidia-smi -i 0,1 -c EXCLUSIVE_PROCESS
CUDA_VISIBLE_DEVICES=0,1 \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
    nvidia-cuda-mps-control -f

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

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
