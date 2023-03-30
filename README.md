# psifx

---

## Installation

### System
```bash
sudo apt install \
  ffmpeg \
  build-essential \
  cmake \
  wget \
  libopenblas-dev \
  libopencv-dev \
  libdlib-dev \
  libboost-all-dev \
  libsqlite3-dev
```

### OpenFace
That will be wrapped into the setup.py, cf. https://stackoverflow.com/questions/47360113/compile-c-library-on-pip-install:
```bash
git clone https://github.com/GuillaumeRochette/OpenFace.git
cd OpenFace
bash download_models.sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=RELEASE ..
make all
```
### Environment
```bash
conda install ffmpeg x264 -c conda-forge # To be able to encode with H264.
conda install pytorch=1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # Then install PyTorch 
```

### Package
Normal install:
```bash
pip install 'git+https://github.com/GuillaumeRochette/psifx.git'
```
Editable install:
```bash
git clone https://github.com/GuillaumeRochette/psifx.git
cd psifx
pip install --editable .
```