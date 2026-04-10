# `psifx` - <u>P</u>sychological and <u>S</u>ocial <u>I</u>nteractions <u>F</u>eature e<u>X</u>traction

---

*`psifx` is a "plug-and-play" multi-modal feature extraction toolkit, aiming to facilitate and democratize the use of state-of-the-art machine learning techniques for human sciences research.
It is motivated by a need 
(a) to automate and standardize data annotation processes, otherwise involving expensive, lengthy, and inconsistent human labor, such as the transcription or coding of behavior changes from audio and video sources;
(b) to develop and distribute open-source community-driven psychology research software;
(c) to enable large-scale access and ease of use to non-expert users.
The framework contains an array of tools for tasks, such as speaker diarization, closed-caption transcription and translation from audio, as well as body, hand, and facial pose estimation and gaze tracking from video.
The package has been designed with a modular and task-oriented approach, enabling the community to add or update new tools easily.
We strongly hope that this package will provide psychologists a simple and practical solution for efficiently a range of audio, linguistic, and visual features from audio and video, thereby creating new opportunities for in-depth study of real-time behavioral phenomena.*

https://github.com/user-attachments/assets/263affe8-f435-42ee-84c9-cabc1f19efde

This demo clip is not intended for commercial use, and is solely for demonstration in an academic or research context.

## Documentation, Reference & Quickstart

Visit https://psifx.github.io/psifx/

arXiv preprint:  https://www.arxiv.org/abs/2407.10266```


# Setup Instructions for psifx with Local SAM3 Model```

## 1. Clone the SAM3 Model Locally

```bash
git clone https://huggingface.co/facebook/sam3
```

Note: Facebook requires ethical approval to download or access the model online. Creating a local copy avoids this requirement and is the simplest solution.

## 2. Clone the psifx Repository

```bash
git clone https://github.com/BogdanvL/psifx.git
```

## 3. Configure SAM3 Path

Edit the following file:

/psifx/utils/constants.py

Update the SAM3_PATH variable to point to your local SAM3 directory:

```python
SAM3_PATH = "/home/[user_name]/path/to/sam3"
```
The default value is:

```python
SAM3_PATH = "facebook/sam3"
```
This requires automatic authorization via Hugging Face.

## 4. Install System Dependencies
```bash
sudo apt install ffmpeg ubuntu-restricted-extras
```
## 5. Create and Activate Conda Environment

```bash
conda create -y -n psifx-env python=3.11 pip
conda activate psifx-env
```
## 6. Install psifx
```bash
cd /home/[user_name]/psifx
pip install .
```
