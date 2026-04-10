# SAM3 Docker Validation Harness

This folder provides a containerized validation flow for SAM3 in `psifx` without requiring a local Conda install.

## What it does
- Builds a fresh Docker image.
- Creates a Conda env (`psifx-env`) inside the image.
- Installs `psifx` from this repository.
- Runs SAM3 CLI smoke tests:
  - `psifx video tracking sam3 inference` on CPU
  - `psifx video tracking sam3 inference` on GPU (if CUDA is available)
  - mask visualization checks for both paths

## Usage

From repository root:

```bash
export HF_TOKEN=your_huggingface_token
bash docker/sam3-validation/run.sh
```

Optional environment variables:

```bash
IMAGE_NAME=psifx-sam3-validation:latest   # Docker image tag
VIDEO_PATH=/workspace/example/data/Video.mp4
USE_GPU=1                                 # Set to 0 for CPU-only container run
```

## Notes
- If `facebook/sam3` is gated for your account, inference will fail unless:
  - `HF_TOKEN` has access to that model, or
  - `SAM3_PATH` is changed to a local model path available inside the container.
- The harness intentionally does not use the repository's top-level `Dockerfile`.

