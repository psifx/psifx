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

ARG USERNAME="root"
ENV HOME="/$USERNAME"
RUN mkdir --parents $HOME $HOME/.config $HOME/.cache && \
    chown --recursive $USERNAME:$USERNAME $HOME && \
    chmod --recursive a+rwx $HOME

# CONDA
ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG CONDA_PREFIX="$HOME/conda"
ENV PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/condabin:${PATH}"
RUN curl -sSf $CONDA_URL -o $HOME/miniconda.sh && \
    bash $HOME/miniconda.sh -b -p $CONDA_PREFIX && \
    rm $HOME/miniconda.sh && \
    conda update -y -c defaults conda && \
    conda install -y python=3.9 pip

# OPENFACE
ARG OPENFACE_PREFIX="$HOME/openface"
ENV PATH="$OPENFACE_PREFIX/build/bin:${PATH}"
RUN wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
    python install.py \
    --license_accepted \
    --install_path $OPENFACE_PREFIX \
    --overwrite_install \
    --minimal_install \
    --no-add_to_login_shell && \
    rm install.py

# HUGGINGFACE
ARG HF_TOKEN
RUN echo "export HF_TOKEN=$HF_TOKEN" >> $HOME/.bashrc

# PSIFX
ARG PSIFX_VERSION
RUN pip install --no-cache-dir git+https://github.com/GuillaumeRochette/psifx.git@$PSIFX_VERSION && \
    chmod --recursive a+rwx $CONDA_PREFIX/lib/python3.9/site-packages/mediapipe