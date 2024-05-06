# Installation

We recommend using Docker for reducing compatibility issues.

## Docker

1. Install [Docker Engine](https://docs.docker.com/engine/install/#server) and make sure to follow
   the [post-install instructions](https://docs.docker.com/engine/install/linux-postinstall/). Otherwise,
   install [Docker Desktop](https://docs.docker.com/desktop/).
2. If you have a GPU and want to use it to accelerate compute:
    1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
    2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
3. Run the latest image:
   ```bash
   export PSIFX_VERSION="X.Y.Z" # Major.Minor.Patch
   export DATA_PATH="/path/to/data" 
   
   docker run \
      --user $(id -u):$(id -g) \
      --gpus all \
      --mount type=bind,source=$DATA_PATH,target=$DATA_PATH \
      --interactive \
      --tty \
      guillaumerochette/psifx:$PSIFX_VERSION
   ```
4. Check out `psifx` available commands!
   ```bash
   psifx --all-help
   ```

## Local

1. Install the following system-wide:
   ```bash
   sudo apt install ffmpeg ubuntu-restricted-extras
   ```
2. Create a dedicated `conda` environment following the instructions in that order:
   ```bash
   conda create -y -n psifx-env python=3.9 pip
   conda activate psifx-env
   ```
3. [Optional] We provide an API endpoint to use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), useable **only** if you
   comply with
   their [license agreement](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt), e.g.
   academic, research or non-commercial purposes.
    1. Install the following system-wide:
       ```bash
       sudo apt install \
       build-essential \
       cmake \
       wget \
       libopenblas-dev \
       libopencv-dev \
       libdlib-dev \
       libboost-all-dev \
       libsqlite3-dev
       ```
    2. Install OpenFace using our fork.
       ```bash
       wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
       python install.py
       ```
4. Install `psifx`:
   ```bash
   pip install 'psifx @ git+https://github.com/GuillaumeRochette/psifx.git'
   ```