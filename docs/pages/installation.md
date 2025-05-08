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
      psifx/psifx:$PSIFX_VERSION
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
3. Install `psifx`:
   ```bash
   pip install 'psifx @ git+https://github.com/psifx/psifx.git'
   ```
   or 
   
   ```bash
   pip install psifx
   ```
   
4. [Optional] We use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) for face inference in videos. The instructions to install OpenFace are in the [Video Processing Guide - Face section](video.md#face).

5. [Optional] Language models are required for the text tools. The installation instructions for each models provider are in the [Text Processing Guide - Model section](text.md#model).