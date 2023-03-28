# psifx

---

## Installation

### System
```bash
sudo apt install ffmpeg
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