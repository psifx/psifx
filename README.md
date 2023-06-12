# `psifx` - Psychological and Social Interactions Feature Extraction

---
## TODO List

- Add `visualization()` for `OpenFaceAnalysisTool`. Implement a similar or simpler version than the one from the original OpenFace.
- Check that every input has the right format, e.g. encoding and extension.
- Re-process the transcribed subtitles with the diarization output.
- Implement the linguistic feature extraction for each of the participant.
- Make two small tools:
  - Audio Conversion from any format to mono wav.
  - Audio Concatenation: Sum of two tracks that are -6dB themselves.
- OpenSMILE
  - Using either diarization or transcription segments:
    - Amplitude windowing, e.g. scalar multiplication of the cos([0, pi/2]) for the first 30ms.
    - Filter with a second order high-pass filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
      - Medium Analytics Vidhya
    - Features level 3, all features.


To extract from a video and split UNIL left and right stereo into two separate mono tracks, downsampled at 16 kHz:
```bash
ffmpeg -i video.mp4 -filter_complex "[0:a]pan=1c|c0=c0[left];[0:a]pan=1c|c0=c1[right]" -map "[left]" -ar 16000 left.wav -map "[right]" -ar 16000 left.wav
```
---
## Setup

---
### Local
To install `psifx` is a Python package, install the following:
- Install the following system-wide:
```bash
sudo apt install ffmpeg ubuntu-restricted-extras
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

