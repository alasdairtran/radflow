FROM nvidia/cuda:10.2-base-ubuntu18.04 AS radflow-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
# build-essential is used to build jsonnet
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
    build-essential \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH

# Enable prompt color in the skeleton .bashrc before creating default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Install conda as qnet and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.8.3 \
    MINICONDA_MD5=d63adf39f2c220950a063e0529d4ff74 \
    CONDA_VERSION=4.8.3
WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes conda && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y

COPY environment.yml /radflow/environment.yml
RUN conda env create -f /radflow/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "radflow", "/bin/bash", "-c"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Apex needs nvcc to compile. We build it in a spearate build to reduce size. #
# Set Docker memory limit to 12GB. An 8GB limit got the apex build killed.    #
# Compute capability: Volta 7.0, Turing 7.5, Ampere 8.0.                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
FROM radflow-base as apex

ENV NCCL_VERSION 2.5.6
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-libraries-$CUDA_PKG_VERSION \
      cuda-nvtx-$CUDA_PKG_VERSION \
      libcublas10=10.2.2.89-1 \
      libnccl2=$NCCL_VERSION-1+cuda10.2 \
      cuda-nvml-dev-$CUDA_PKG_VERSION \
      cuda-command-line-tools-$CUDA_PKG_VERSION \
      cuda-libraries-dev-$CUDA_PKG_VERSION \
      cuda-minimal-build-$CUDA_PKG_VERSION \
      libnccl-dev=$NCCL_VERSION-1+cuda10.2 \
      libcublas-dev=10.2.2.89-1 && \
      libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
    apt-mark hold libnccl2 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nvidia/apex \
 && cd apex && git checkout 4ef930c1c884f \
 && TORCH_CUDA_ARCH_LIST="7.0;7.5" pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Adding apex to our main image                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
FROM radflow-base

COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/apex /opt/conda/envs/radflow/lib/python3.8/site-packages/apex
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/apex-0.1-py3.8.egg-info /opt/conda/envs/radflow/lib/python3.8/site-packages/apex-0.1-py3.8.egg-info
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/mlp_cuda.cpython-38-x86_64-linux-gnu.so /opt/conda/envs/radflow/lib/python3.8/site-packages/mlp_cuda.cpython-38-x86_64-linux-gnu.so
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/syncbn.cpython-38-x86_64-linux-gnu.so /opt/conda/envs/radflow/lib/python3.8/site-packages/syncbn.cpython-38-x86_64-linux-gnu.so
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/fused_layer_norm_cuda.cpython-38-x86_64-linux-gnu.so /opt/conda/envs/radflow/lib/python3.8/site-packages/fused_layer_norm_cuda.cpython-38-x86_64-linux-gnu.so
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/amp_C.cpython-38-x86_64-linux-gnu.so /opt/conda/envs/radflow/lib/python3.8/site-packages/amp_C.cpython-38-x86_64-linux-gnu.so
COPY --from=apex /opt/conda/envs/radflow/lib/python3.8/site-packages/apex_C.cpython-38-x86_64-linux-gnu.so /opt/conda/envs/radflow/lib/python3.8/site-packages/apex_C.cpython-38-x86_64-linux-gnu.so

# jina requires docker
RUN pip install --no-cache-dir -U torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html && \
    pip install --no-cache-dir -U torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html && \
    pip install --no-cache-dir -U torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html && \
    pip install --no-cache-dir -U torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html && \
    pip install --no-cache-dir -U torch-geometric && \
    pip install --no-cache-dir -U docker

# conda run, although more correct, buffers stdout and nothing is shown
ENV PATH /opt/conda/envs/radflow/bin:$PATH
# ENTRYPOINT ["conda", "run", "-n", "radflow"]

COPY . /radflow
WORKDIR /radflow
RUN cd /radflow && python setup.py develop

# Remove when this is merged: https://github.com/jina-ai/jina/issues/924
RUN sed -i 's/args.uses/args.yaml_path/g' /opt/conda/envs/radflow/lib/python3.8/site-packages/jina/main/api.py
RUN sed -i '84d' /opt/conda/envs/radflow/lib/python3.8/site-packages/jina/main/api.py

ENTRYPOINT ["jina", "flow", "--yaml-path", "jina/flows/query.yaml"]
