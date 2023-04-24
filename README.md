# `psifx` - Psychological and Social Interactions Feature Extraction

---

## Setup

### Local
To install `psifx` is a Python package, install the following:
- Install the following system-wide:
```bash
sudo apt install ffmpeg
sudo apt install ubuntu-restricted-extras
```
- Create a dedicated `conda` environment following the instructions in that order:
```bash
CONDA_ENV_NAME="name-of-my-environment"
conda create -y -n $CONDA_ENV_NAME python=3.9 pip
conda activate $CONDA_ENV_NAME
conda install -y ffmpeg x264 -c conda-forge # To be able to encode with H264.
conda install -y pytorch=1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # Then install PyTorch 
```
- Now install `psifx`:
```bash
# For now:
pip install 'git+https://github.com/GuillaumeRochette/psifx.git'
# Once we release it publicly:
pip install psifx 
# For an editable install:
git clone https://github.com/GuillaumeRochette/psifx.git
cd psifx
pip install --editable .
```
- We provide an API endpoint to use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), if you comply with their [license agreement](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt), e.g. academic, research or non-commercial purposes, then you can install our fork:
```bash
# Install these system-wide.
sudo apt install \
  build-essential \
  cmake \
  wget \
  libopenblas-dev \
  libopencv-dev \
  libdlib-dev \
  libboost-all-dev \
  libsqlite3-dev
# Download the install.py
wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py
# Run the interactive install.py
python install.py
```

### Docker
We provide a Docker image for convenience, or distributed usage:  
```bash
# Add docker images.
```

## Usage

### As a Python package

### As a Command Line Interface

