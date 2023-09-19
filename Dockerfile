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
    ubuntu-restricted-extras \
    unzip \
    wget && \
    apt-get -y autoremove && \
    apt-get -y autoclean  && \
    apt-get -y clean

# CONDA
ENV CONDA_PREFIX="/opt/conda"
ENV PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/condabin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_PREFIX && \
    rm miniconda.sh
RUN conda update -n base -c defaults conda

# OPENFACE
ENV OPENFACE_PREFIX="/opt/openface"
ENV PATH="$OPENFACE_PREFIX/build/bin:${PATH}"
RUN wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
    python install.py \
    --license_accepted \
    --install_path $OPENFACE_PREFIX \
    --overwrite_install \
    --minimal_install \
    --no-add_to_login_shell && \
    rm install.py

# PSIFX
RUN pip install 'git+https://github.com/GuillaumeRochette/psifx.git'

# CREATE DIRECTORIES WHERE MODEL CHECKPOINTS WILL BE DOWNLOADED BY ANY USERS
RUN mkdir --mode 777 \
    /.config \
    /.cache && \
    chmod --recursive 777 /opt/conda
