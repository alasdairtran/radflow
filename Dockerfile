FROM nvidia/cuda:10.2-base-ubuntu18.04

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

COPY . /radflow
RUN conda env update -f /radflow/environment.yml && \
    conda activate radflow && \
    cd /radflow && python setup.py install \
    pip install -U torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    pip install -U torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    pip install -U torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    pip install -U torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    pip install -U torch-geometric

WORKDIR /radflow
