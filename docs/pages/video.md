# Video Processing Guide

This document provides instructions for using the `psifx` package for video inference and visualization, including
manipulation, pose, and face.

## Manipulation

### Process Video

Extract a specific time window from a video, crop it based on a bounding box, and optionally rescale the video.

```bash
psifx video manipulation process \
    --in_video Video.mp4 \
    --out_video VideoProcessed.mp4 \
    --start 18 \
    --end 210 \
    --x_min 1347 \
    --y_min 459 \
    --x_max 2553 \
    --y_max 1898 \
    [--width 1920] \
    [--height 1080]
```

- `--in_video`: Path to the input video file, such as `/path/to/video.mp4` (or `.avi`, `.mkv`, etc.).
- `--out_video`: Path to the output video file, such as `/path/to/video.mp4` (or `.avi`, `.mkv`, etc.).
- `--start`: Timestamp in seconds for the start of the selection.
- `--end`: Timestamp in seconds for the end of the selection.
- `--x_min`: X-axis coordinate of the top-left corner for cropping in pixels.
- `--y_min`: Y-axis coordinate of the top-left corner for cropping in pixels.
- `--x_max`: X-axis coordinate of the bottom-right corner for cropping in pixels.
- `--y_max`: Y-axis coordinate of the bottom-right corner for cropping in pixels.
- `--width`: Width of the resized output video, default `None`.
- `--height`: Height of the resized output video, default `None`.

## Tracking

Segment and track humans/objects in a video using **Yolo** and **Samurai**.

### Tracking with Samurai

Detect the specified humans or objects with **YOLO**, and track them across the video with Samurai (**SAM-2** model).
To work, it is necessary to have at least an instant in the video where all humans/objects of interest appear
simultaneously. For detection, this method goes through the whole video to find the instant with the maximum number of
humans/objects of interest.
It stops early if all objects are detected, as specified by `max_objects`. The id assignment (1 up to the number of
detected humans/objects) are arbitrary.

```bash
psifx video tracking samurai inference \
    --video Video.mp4 \
    --mask_dir MaskDir \
    [--model_size tiny] \
    [--yolo_model yolo11n.pt] \
    [--object_class 0] \
    [--max_objects 100] \
    [--step 30] \
    [--device cuda] \
```

* `--video`: Path to the input video file (supports `.mp4`, `.avi`, `.mkv`, etc.).
* `--mask_dir`: Path to the output mask directory.
* `--model_size`: Size of the SAM-2 model, either `tiny`, `small`, `base_plus`, or `large` default is `tiny`.
* `--yolo_model`: Name of the YOLO model to use, from small to big: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`,
  `yolo11l.pt`, `yolo11x.pt`, default is `yolo11n.pt`.
* `--object_class`: Class of the object to detect from `0` (_people_) to `79`(_toothbrush_), default is
  `0`. [Click here for the full list.](https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8)
* `--max_objects`: Maximum number of people/objects to detect. If not specified, the method will search for the frame
  with the most people/objects.
* `--step`: Step size in frames to perform object detection, default is `30`.
* `--device`: Device on which to run the inference, either `cpu` or `cuda`, default is `cpu`.

### Tracking Visualization

Creates a visual video representation of the tracked masks, with optional blackout, labels, and coloring.

```bash
psifx video tracking visualization \
    --video Video.mp4 \
    --masks Mask1.mp4 Mask2.mp4 MaskDir \
    --visualization Visualization.mp4 \
    [--blackout False] \
    [--labels True] \
    [--color True] \
```

* `--video`: Path to the input video file (supports `.mp4`, `.avi`, `.mkv`, etc.).
* `--masks`: List of paths to mask directories or individual `.mp4` mask files.
* `--visualization`: Path to the output visualization video file.
* `--blackout`: Whether to black out the background (non-mask regions), default is `False`.
* `--labels`: Whether to add labels to the visualized objects, default is `True`.
* `--color`: Whether to color the masks for better visual distinction, default is `True`.

## Pose

### Pose Estimation

Detects and analyzes human poses in a video using MediaPipe.

There are two methods for whether the video contains a single person or multiple people.

#### Single-Inference

```bash
psifx video pose mediapipe single-inference \
    --video Video.mp4 \
    --poses Poses.tar.gz \
    [--mask Mask.mp4] \
```

- `--video`: Input video file for pose estimation, can be `.mp4`, `.avi`, `.mkv`, etc...
- `--poses`: Path to save pose estimation data in `.tar.gz` format.
- `--mask`: Path to an optional input `.mp4` mask file.

#### Multi-Inference

Multi-inference requires the masks which are generated with the samurai tracking tool.

```bash
psifx video pose mediapipe multi-inference \
    --video Video.mp4 \
    --masks Mask1.mp4 Mask2.mp4 MaskDir \
    --poses_dir PosesDir \
```

- `--video`: Input video file for pose estimation, can be `.mp4`, `.avi`, `.mkv`, etc...
- `--masks`: List of path to mask directories or individual `.mp4` mask files.
- `--poses_dir`: Directory path to save pose estimation data.

#### Common Optional Arguments

These arguments can be used in both the above commands to configure the inference.

```bash
    [--mask_threshold 0.1] \
    [--model_complexity 2] \
    [--smooth] \
    [--device cuda]
```

- `--mask_threshold`: Threshold for the binarization of the segmentation mask, default `0.1`.
- `--model_complexity`: Complexity of the model: {`0`, `1`, `2`}, higher means more FLOPs, but also more accurate
  results, default `2`.
- `--smooth`: Temporally smooth the inference results to reduce the jitter, default `True`.
- `--device`: Device on which to run the inference, either `cpu` or `cuda`, by default `cuda` if available.

### Pose Visualization

Creates a visual representation of detected poses in the video.

```bash
psifx video pose mediapipe visualization \
    --video Video.mp4 \
    --poses Pose1.tar.gz Pose2.tar.gz PoseDir \
    --visualization Visualization.mediapipe.mp4 \
    [--confidence_threshold 0.0]
```

- `--video`: Original video file for pose visualization overlay.
- `--poses`: List of path to the input pose directories or individual archive ``.tar.gz`` files.
- `--visualization`: Path to save the visualized output video.
- `--confidence_threshold`: Threshold for not displaying low confidence keypoints, default `0.0`.

## Face

### Requirements

The face feature extraction requires [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).
We provide an API endpoint to use OpenFace, useable **only** if you
comply with their [license agreement](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt),
e.g.
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

There are two method for whether the video contains a single person or multiple people.

#### Single-Inference

```bash
psifx video face openface single-inference \
    --video Video.mp4 \
    --features Faces.tar.gz \
    [--mask Mask.mp4] \
    [--device cuda]
```

- `--video`: Input video file for face feature extraction.
- `--features`: Path to save extracted facial features in `.tar.gz` format.
- `--mask`: Path to an optional input .mp4 mask file.
- `--device`: Device on which to run the inference, either `cpu` or `cuda`, by default `cuda` if available.

#### Multi-Inference

To perform multi-inference, it is necessary to get the masks from the tracking tool beforehand.

```bash
psifx video face openface multi-inference \
    --video Video.mp4 \
    --masks Mask1.mp4 Mask2.mp4 MaskDir \
    --features_dir FacesDir \
    [--device cuda]
```

- `--video`: Input video file for face feature extraction.
- `--masks`: List of path to mask directories or individual .mp4 mask files.
- `--features_dir`: Directory path to save extracted facial features.
- `--device`: Device on which to run the inference, either `cpu` or `cuda`, by default `cuda` if available.

### Face Visualization

Creates a visual overlay of facial features detected in the video.

```bash
psifx video face openface visualization \
    --video Video.mp4 \
    --features Faces1.tar.gz Faces2.tar.gz FacesDir \
    --visualization Visualization.openface.mp4 \
    [--depth 3.0] \
    [--f_x 1600.0] [--f_y 1600.0] \
    [--c_x 960.0] [--c_y 540.0]
```

- `--video`: Original video file for face feature visualization overlay.
- `--features`: List of path to the input facial feature directories or individual archive ``.tar.gz`` files.
- `--visualization`: Path to save the visualized output video.
- `--depth`: Projection: assumed static depth of the subject in meters, default `3.0`.
- `--f_x`, `--f_y`: Projection: x-axis (respectively y-axis) of the focal length, default `None`.
- `--c_x`, `--c_y`: Projection: x-axis (respectively y-axis) of the principal point, default `None`.