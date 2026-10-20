# Installation

We recommend using Docker for reducing compatibility issues.

## Docker

1. Install [Docker Engine](https://docs.docker.com/engine/install/#server) and make sure to follow
   the [post-install instructions](https://docs.docker.com/engine/install/linux-postinstall/). Otherwise,
   install [Docker Desktop](https://docs.docker.com/desktop/).
2. If you have a GPU and want to use it to accelerate compute:
    1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
    2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
3. For SAM3, request access to [facebook/sam3](https://huggingface.co/facebook/sam3), then create a
   Hugging Face read token from that account. Store your own token in a local `.env` file as
   `HF_TOKEN=...`. This file is supplied at runtime and is not included in the image.
4. Run a published image (0.2.4 or newer includes OSNet identity correction dependencies):
   ```bash
   export PSIFX_VERSION="0.2.4" # Available after the 0.2.4 release is published
   export DATA_PATH="/path/to/data" 
   
   docker run \
      --user $(id -u):$(id -g) \
      --gpus all \
      --env-file .env \
      --mount type=bind,source=$DATA_PATH,target=$DATA_PATH \
      --interactive \
      --tty \
      psifx/psifx:$PSIFX_VERSION
   ```
   Omit `--gpus all` for CPU-only use. The first SAM3 and OSNet runs download model weights;
   reuse writable Hugging Face and Torch caches for subsequent runs.
5. Check out `psifx` available commands!
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
   conda create -y -n psifx-env python=3.11 pip
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
   For SAM3 identity correction with OSNet, install the optional extra:
   ```bash
   pip install 'psifx[reid]'
   ```
   For a source install, use `pip install '.[reid]'` from the checkout.
   Without the extra, correction warns and falls back to color-only matching.

4. [Optional] We use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) for face inference in videos. The instructions to install OpenFace are in the [Video Processing Guide - Face section](video.md#face).

5. [Optional] Language models are required for the text tools. The installation instructions for each models provider are in the [Text Processing Guide - Model section](text.md#model).

6. For SAM3, supply your own model-approved Hugging Face token through `HF_TOKEN` or `--api_token`.
   The CLI does not prompt or automatically load `.env`. To load a trusted local `.env` in Bash:
   ```bash
   set -a
   source .env
   set +a
   ```
   A complete downloaded model can instead be passed with `--model_path`.
   See the [SAM3 guide](video.md#tracking-with-sam3) for output directories, options, and limitations.
