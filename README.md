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

arXiv preprint: https://www.arxiv.org/abs/2407.10266

## SAM3 tracking quickstart

Install FFmpeg and create a Python environment:

```bash
sudo apt install ffmpeg ubuntu-restricted-extras
conda create -y -n psifx-env python=3.11 pip
conda activate psifx-env
git clone https://github.com/psifx/psifx.git
cd psifx
pip install '.[reid]'
```

For a published version, use `pip install 'psifx[reid]'`. The `reid` extra enables OSNet appearance
embeddings for identity correction; Docker images from 0.2.4 include it.

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) and create your own
[Hugging Face read token](https://huggingface.co/docs/hub/security-tokens) with access to that model.
Supply it through the `HF_TOKEN` environment variable or `--api_token`. The CLI does not prompt
interactively or automatically read `.env`; Docker can load your own token file using `--env-file .env`.
If you already downloaded a complete model directory, use `--model_path /path/to/sam3`
or set `SAM3_PATH`. Downloading a local copy still requires the model's access approval.

```bash
psifx video tracking sam3 inference \
    --video example/data/Video.mp4 \
    --mask_dir example/data/masks \
    --device cuda \
    --chunk_size 64

psifx video tracking visualization \
    --video example/data/Video.mp4 \
    --masks example/data/masks_merged \
    --visualization example/data/tracking_corrected.mp4
```

Use `--device cpu` if CUDA is unavailable. Tracking writes raw masks to `masks`, corrected masks to
`masks_merged`, and decisions to `masks_merge_report.json`. Correction is enabled by default; use
`--no-id_correction` to disable it. Review the overlays and report before using corrected identities:
adjacent people can be merged incorrectly, and border-touching or changing appearances can remain split.
OSNet downloads weights on first use; without the optional dependencies, correction falls back to color.

See the [video guide](docs/pages/video.md), [installation guide](docs/pages/installation.md),
and [worked example](example/README.md) for all options and downstream pose extraction.
