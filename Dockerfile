FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && \
    apt-get -y install \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        htop \
        libboost-all-dev \
        libdlib-dev \
        libopenblas-dev \
        libopencv-dev \
        libopencv-calib3d-dev \
        libopencv-contrib-dev \
        libopencv-features2d-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        libsqlite3-dev \
        nano \
        ninja-build \
        software-properties-common \
        sudo \
        ubuntu-restricted-extras \
        unzip \
        wget && \
    apt-get -y autoremove && \
    apt-get -y autoclean  && \
    apt-get -y clean

ARG PYTHON_VERSION=3.9
ARG HF_TOKEN

ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG USERNAME="root"
ENV HOME="/$USERNAME"
ARG CONDA_PREFIX="$HOME/conda"
ARG OPENFACE_PREFIX="$HOME/openface"
ARG PSIFX_PREFIX="$HOME/psifx"
ENV PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/condabin:$OPENFACE_PREFIX/build/bin:${PATH}"
ENV HF_TOKEN=$HF_TOKEN

COPY . $PSIFX_PREFIX
RUN mkdir --parents $HOME $HOME/.config $HOME/.cache && \
    chown --recursive $USERNAME:$USERNAME $HOME && \
    chmod --recursive a+rwx $HOME && \
    curl -sSf $CONDA_URL -o $HOME/miniconda.sh && \
    bash $HOME/miniconda.sh -b -p $CONDA_PREFIX && \
    rm $HOME/miniconda.sh && \
    conda update -y -c defaults conda && \
    conda install -y python=$PYTHON_VERSION pip && \
    pip cache purge && \
    conda clean -y --all && \
    wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
    python install.py \
        --license_accepted \
        --install_path $OPENFACE_PREFIX \
        --overwrite_install \
        --minimal_install \
        --no-add_to_login_shell && \
    rm install.py && \
    pip install --no-cache-dir $PSIFX_PREFIX && \
    rm -r $PSIFX_PREFIX && \
    pip cache purge && \
    conda clean -y --all && \
    chmod --recursive a+rwx $CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/mediapipe && \
    curl -fsSL https://ollama.com/install.sh | sh
