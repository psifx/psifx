# Video Processing Guide

This document provides instructions for using the `psifx` package for video inference and visualization, including pose and face.


## Pose

### Pose Estimation
Detects and analyzes human poses in a video using MediaPipe.
```bash
psifx video pose mediapipe inference \
    --video Video.mp4 \
    --poses Poses.tar.xz \
    [--masks Masks.mp4] \
    [--mask_threshold 0.1] \
    [--model_complexity 2] \
    [--smooth] \
    [--device cuda]
```
- `--video`: Input video file for pose estimation, can be `.mp4`, `.avi`, `.mkv`, etc...
- `--poses`: Path to save pose estimation data in `.tar.xz` format.
- `--masks`: Path to save a video mask showing detected body parts.
- `--mask_threshold`: Threshold for the binarization of the segmentation mask, default `0.1`.
- `--model_complexity`: Complexity of the model: {`0`, `1`, `2`}, higher means more FLOPs, but also more accurate results, default `2`.
- `--smooth`: Temporally smooth the inference results to reduce the jitter, default `True`.
- `--device`: Device on which to run the inference, either `cpu` or `cuda`, default `cpu`.

### Pose Visualization
Creates a visual representation of detected poses in the video.
```bash
psifx video pose mediapipe visualization \
    --video Video.mp4 \
    --poses Poses.tar.xz \
    --visualization Visualization.mediapipe.mp4 \
    [--confidence_threshold 0.0]
```
- `--video`: Original video file for pose visualization overlay.
- `--poses`: Pose estimation data file in `.tar.xz` format.
- `--visualization`: Path to save the visualized output video.
- `--confidence_threshold`: Threshold for not displaying low confidence keypoints, default `0.0`.

## Face

### Requirements
The face feature extraction requires [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). 
We provide an API endpoint to use OpenFace, useable **only** if you
   comply with their [license agreement](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt), e.g.
   academic, research or non-commercial purposes.
1. Install the following system-wide:
   ```bash
   sudo apt install \
   build-essential \
   cmake \
   wget \
   libopenblas-dev \
   libopencv-dev \
   libdlib-dev \
   libboost-all-dev \
   libsqlite3-dev
   ```
2. Install OpenFace using our fork.
   ```bash
   wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
   python install.py
   ```


### Face Feature Extraction
Extracts facial features from a video using OpenFace.
To use OpenFace you need to install it beforehand follow the installation instructions.
```bash
psifx video face openface inference \
    --video Video.mp4 \
    --features Faces.tar.xz
```
- `--video`: Input video file for face feature extraction.
- `--features`: Path to save extracted facial features in `.tar.xz` format.


### Face Visualization
Creates a visual overlay of facial features detected in the video.
```bash
psifx video face openface visualization \
    --video Video.mp4 \
    --features Faces.tar.xz \
    --visualization Visualization.openface.mp4 \
    [--depth 3.0] \
    [--f_x 30.0] [--f_y 30.0] \
    [--c_x 0.0] [--c_y 0.0] 
```
- `--video`: Original video file for face feature visualization overlay.
- `--features`: Extracted facial feature data file in `.tar.xz` format.
- `--visualization`: Path to save the visualized output video.
- `--depth`: Projection: assumed static depth of the subject in meters, default `3.0`.
- `--f_x`, `--f_y`: Projection: x-axis (respectively y-axis) of the focal length, default `None`.
- `--c_x`, `--c_y`: Projection: x-axis (respectively y-axis) of the principal point, default `None`.

