# Setup

---
## Package Side

System:
```bash
sudo apt install ffmpeg
```
Environment:
```bash
conda install ffmpeg x264 -c conda-forge # To be able to encode with H264.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # Then install PyTorch 
```

```bash
pip install ffmpeg-python mediapipe scikit-video tqdm 'git+https://github.com/pyannote/pyannote-audio.git@b56add2' openai-whisper pydub
```

## Dev Side

Environment:
```bash
pip install --upgrade setuptools build
```
Build:
```bash
 python -m build
```
Shallow Install:
```bash
pip install --editable .
```
