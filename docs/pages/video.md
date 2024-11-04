# Video Processing Guide

This document provides instructions for using the `psifx` package for video inference and visualization.

## 1. Inference

### Pose Estimation
Detects and analyzes human poses in a video using MediaPipe.
```bash
psifx video pose mediapipe inference \
    --video Videos/Right.mp4 \
    --poses Poses/Right.tar.xz \
    --masks Masks/Right.mp4
```
- `--video`: Input video file for pose estimation.
- `--poses`: Path to save pose estimation data in `.tar.xz` format.
- `--masks`: Path to save a video mask showing detected body parts.

Repeat the command for other videos as needed:
```bash
psifx video pose mediapipe inference \
    --video Videos/Left.mp4 \
    --poses Poses/Left.tar.xz \
    --masks Masks/Left.mp4
```

### Face Feature Extraction
Extracts facial features from a video using OpenFace.
```bash
psifx video face openface inference \
    --video Videos/Right.mp4 \
    --features Faces/Right.tar.xz
```
- `--video`: Input video file for face feature extraction.
- `--features`: Path to save extracted facial features in `.tar.xz` format.

Repeat the command for other videos as needed:
```bash
psifx video face openface inference \
    --video Videos/Left.mp4 \
    --features Faces/Left.tar.xz
```

## 2. Visualization

### Pose Visualization
Creates a visual representation of detected poses in the video.
```bash
psifx video pose mediapipe visualization \
    --video Videos/Right.mp4 \
    --poses Poses/Right.tar.xz \
    --visualization Visualizations/Right.mediapipe.mp4
```
- `--video`: Original video file for pose visualization overlay.
- `--poses`: Pose estimation data file.
- `--visualization`: Path to save the visualized output video.

Repeat the command for other videos as needed:
```bash
psifx video pose mediapipe visualization \
    --video Videos/Left.mp4 \
    --poses Poses/Left.tar.xz \
    --visualization Visualizations/Left.mediapipe.mp4
```

### Face Visualization
Creates a visual overlay of facial features detected in the video.
```bash
psifx video face openface visualization \
    --video Videos/Right.mp4 \
    --features Faces/Right.tar.xz \
    --visualization Visualizations/Right.openface.mp4
```
- `--video`: Original video file for face feature visualization overlay.
- `--features`: Extracted facial feature data file.
- `--visualization`: Path to save the visualized output video.

Repeat the command for other videos as needed:
```bash
psifx video face openface visualization \
    --video Videos/Left.mp4 \
    --features Faces/Left.tar.xz \
    --visualization Visualizations/Left.openface.mp4
```