FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n psifx-env python=${PYTHON_VERSION} pip && \
    conda clean -afy

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace
COPY . /workspace

RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate psifx-env && \
    pip install --upgrade pip setuptools wheel && \
    pip install .

RUN chmod +x /workspace/docker/sam3-validation/validate_sam3.sh
