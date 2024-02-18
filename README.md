# `psifx` - Psychological and Social Interactions Feature Extraction

---

## Content

1. [Installation](#installation)
   1. [Docker](#docker)
   2. [Local](#local)
2. [Usage](#usage)
   1. [Concept](#concept)
   2. [Audio](#audio)
      1. [Manipulation](#manipulation)
         1. [Extraction](#extraction)
         2. [Conversion](#conversion)
         3. [Mixdown](#mixdown)
         4. [Normalization](#normalization)
      2. [Diarization](#diarization)
         1. [Inference](#inference)
         2. [Visualization](#visualization)
      3. [Identification](#identification)
         1. [Inference](#inference-1)
      4. [Transcription](#transcription)
         1. [Inference](#inference-2)
         2. [Enhance](#enhance)
      5. [Non-verbal Feature Extraction](#non-verbal-feature-extraction)
         1. [Inference](#inference-3)
   3. [Video](#video)
      1. [Manipulation](#manipulation-1)
         1. [Process](#process)
      2. [Pose Estimation](#pose-estimation)
         1. [Inference](#inference-4)
         2. [Visualization](#visualization-1)
      3. [Face Analysis](#face-analysis)
         1. [Inference](#inference-5)
         2. [Visualization](#visualization-2)
3. [Examples](#examples)
---

## Installation

We recommend using Docker for reducing compatibility issues.

### Docker

1. Install [Docker Engine](https://docs.docker.com/engine/install/#server) and make sure to follow
   the [post-install instructions](https://docs.docker.com/engine/install/linux-postinstall/). Otherwise,
   install [Docker Desktop](https://docs.docker.com/desktop/).
2. If you have a GPU and want to use it to accelerate compute:
    1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
    2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
3. Run the latest image:
   ```bash
   export PSIFX_VERSION="0.0.2"
   
   docker run \
      --user $(id -u):$(id -g) \
      --gpus all \
      --mount type=bind,source=/path/to/data,target=/path/to/data \
      --interactive \
      --tty \
      guillaumerochette/psifx:$PSIFX_VERSION
   ```
4. Check out `psifx` available commands!
   ```bash
   psifx --all-help
   ```

### Local

1. Install the following system-wide:
   ```bash
   sudo apt install ffmpeg ubuntu-restricted-extras
   ```
2. Create a dedicated `conda` environment following the instructions in that order:
   ```bash
   conda create -y -n psifx-env python=3.9 pip
   conda activate psifx-env
   ```
3. Now install `psifx`:
   ```bash
   pip install 'git+https://github.com/GuillaumeRochette/psifx.git'
   ```
4. We provide an API endpoint to use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), useable only if you
   comply with
   their [license agreement](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt), e.g.
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

---

## Usage

### Concept

`psifx` is a Python package that can be used both as a library,
```python
from psifx.audio.diarization.pyannote.tool import PyannoteDiarizationTool

# Parameterize a tool w/ specific settings, such as choosing the underlying neural network, etc.
tool = PyannoteDiarizationTool(...)
# Run the inference method on a given data, here it will be an audio track for example.
tool.inference(...)
```
But it can also come with its own CLI, that can be run directly in a terminal,
```bash
psifx audio diarization pyannote inference --audio /path/to/audio.wav --diarization /path/to/diarization.rttm
```

### Audio

#### Manipulation

##### Extraction

```
psifx audio manipulation extraction [-h] --video VIDEO --audio AUDIO
                                           [--overwrite | --no-overwrite]
                                           [--verbose | --no-verbose]

    Tool for extracting the audio track from a video.

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --audio AUDIO         path to the output audio file, such as '/path/to/audio.wav'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Conversion

```
psifx audio manipulation conversion [-h] --audio AUDIO --mono_audio
                                           MONO_AUDIO
                                           [--overwrite | --no-overwrite]
                                           [--verbose | --no-verbose]

    Tool for converting any audio track to a mono audio track at 16kHz sample rate.

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         path to the input audio file, such as '/path/to/audio.wav' (or .mp3, etc.)
  --mono_audio MONO_AUDIO
                        path to the output audio file, such as '/path/to/mono-audio.wav'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Mixdown

```
psifx audio manipulation mixdown [-h] --mono_audios MONO_AUDIOS
                                        [MONO_AUDIOS ...] --mixed_audio
                                        MIXED_AUDIO
                                        [--overwrite | --no-overwrite]
                                        [--verbose | --no-verbose]

    Tool for mixing multiple mono audio tracks.

optional arguments:
  -h, --help            show this help message and exit
  --mono_audios MONO_AUDIOS [MONO_AUDIOS ...]
                        paths to the input mono audio files, such as '/path/to/mono-audio-1.wav /path/to/mono-audio-2.wav'
  --mixed_audio MIXED_AUDIO
                        path to the output mixed audio file, such as '/path/to/mixed-audio.wav'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Normalization

```
psifx audio manipulation normalization [-h] --audio AUDIO
                                              --normalized_audio
                                              NORMALIZED_AUDIO
                                              [--overwrite | --no-overwrite]
                                              [--verbose | --no-verbose]

    Tool for normalizing an audio track.

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         path to the input audio file, such as '/path/to/audio.wav'
  --normalized_audio NORMALIZED_AUDIO
                        path to the output normalized audio file, such as '/path/to/normalized-audio.wav'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```


#### Diarization

##### Inference

```
psifx audio diarization pyannote inference [-h] --audio AUDIO
                                                  --diarization DIARIZATION
                                                  [--num_speakers NUM_SPEAKERS]
                                                  [--model_name MODEL_NAME]
                                                  [--api_token API_TOKEN]
                                                  [--device DEVICE]
                                                  [--overwrite | --no-overwrite]
                                                  [--verbose | --no-verbose]

    Tool for diarizing an audio track with pyannote.

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         path to the input audio file, such as '/path/to/audio.wav'
  --diarization DIARIZATION
                        path to the output diarization file, such as '/path/to/diarization.rttm'
  --num_speakers NUM_SPEAKERS
                        number of speaking participants, if ignored the model will try to guess it, it is advised to specify it
  --model_name MODEL_NAME
                        version number of the pyannote/speaker-diarization model, c.f. https://huggingface.co/pyannote/speaker-diarization/tree/main/reproducible_research
  --api_token API_TOKEN
                        API token for the downloading the models from HuggingFace
  --device DEVICE       device on which to run the inference, either 'cpu' or 'cuda'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Visualization

```
psifx audio diarization pyannote visualization [-h] --diarization
                                                      DIARIZATION
                                                      --visualization
                                                      VISUALIZATION
                                                      [--overwrite | --no-overwrite]
                                                      [--verbose | --no-verbose]

    Tool for visualizing the diarization of a track.

optional arguments:
  -h, --help            show this help message and exit
  --diarization DIARIZATION
                        path to the input diarization file, such as '/path/to/diarization.rttm'
  --visualization VISUALIZATION
                        path to the output visualization file, such as '/path/to/visualization.png'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

#### Identification

##### Inference

```
psifx audio identification pyannote inference [-h] --mixed_audio
                                                     MIXED_AUDIO --diarization
                                                     DIARIZATION --mono_audios
                                                     MONO_AUDIOS
                                                     [MONO_AUDIOS ...]
                                                     --identification
                                                     IDENTIFICATION
                                                     [--model_names MODEL_NAMES [MODEL_NAMES ...]]
                                                     [--api_token API_TOKEN]
                                                     [--device DEVICE]
                                                     [--overwrite | --no-overwrite]
                                                     [--verbose | --no-verbose]

    Tool for identifying speakers from an audio track with pyannote.

optional arguments:
  -h, --help            show this help message and exit
  --mixed_audio MIXED_AUDIO
                        path to the input mixed audio file, such as '/path/to/mixed-audio.wav'
  --diarization DIARIZATION
                        path to the input diarization file, such as '/path/to/diarization.rttm'
  --mono_audios MONO_AUDIOS [MONO_AUDIOS ...]
                        paths to the input mono audio files, such as '/path/to/mono-audio-1.wav /path/to/mono-audio-2.wav'
  --identification IDENTIFICATION
                        path to the output identification file, such as '/path/to/identification.json'
  --model_names MODEL_NAMES [MODEL_NAMES ...]
                        names of the embedding models
  --api_token API_TOKEN
                        API token for the downloading the models from HuggingFace
  --device DEVICE       device on which to run the inference, either 'cpu' or 'cuda'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

#### Transcription

##### Inference

```
psifx audio transcription whisper inference [-h] --audio AUDIO
                                                   --transcription
                                                   TRANSCRIPTION
                                                   [--language LANGUAGE]
                                                   [--model_name MODEL_NAME]
                                                   [--translate_to_english | --no-translate_to_english]
                                                   [--device DEVICE]
                                                   [--overwrite | --no-overwrite]
                                                   [--verbose | --no-verbose]

    Tool for transcribing an audio track with Whisper.

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         path to the input audio file, such as '/path/to/audio.wav'
  --transcription TRANSCRIPTION
                        path to the output transcription file, such as '/path/to/transcription.vtt'
  --language LANGUAGE   language of the audio, if ignore, the model will try to guess it, it is advised to specify it
  --model_name MODEL_NAME
                        name of the model, check https://github.com/openai/whisper#available-models-and-languages
  --translate_to_english, --no-translate_to_english
                        whether to transcribe the audio in its original language or to translate it to english (default: False)
  --device DEVICE       device on which to run the inference, either 'cpu' or 'cuda'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Enhance

```
psifx audio transcription whisper enhance [-h] --transcription
                                                 TRANSCRIPTION --diarization
                                                 DIARIZATION --identification
                                                 IDENTIFICATION
                                                 --enhanced_transcription
                                                 ENHANCED_TRANSCRIPTION
                                                 [--overwrite | --no-overwrite]
                                                 [--verbose | --no-verbose]

    Tool for enhancing a transcription with diarization and identification.

optional arguments:
  -h, --help            show this help message and exit
  --transcription TRANSCRIPTION
                        path to the input transcription file, such as '/path/to/transcription.vtt'
  --diarization DIARIZATION
                        path to the input diarization file, such as '/path/to/diarization.rttm'
  --identification IDENTIFICATION
                        path to the input identification file, such as '/path/to/identification.json'
  --enhanced_transcription ENHANCED_TRANSCRIPTION
                        path to the output transcription file, such as '/path/to/enhanced-transcription.vtt'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

#### Non-verbal Feature Extraction

##### Inference

```
psifx audio speech opensmile inference [-h] --audio AUDIO --diarization
                                              DIARIZATION --features FEATURES
                                              [--feature_set FEATURE_SET]
                                              [--feature_level FEATURE_LEVEL]
                                              [--overwrite | --no-overwrite]
                                              [--verbose | --no-verbose]

    Tool for extracting non-verbal speech features from an audio track with OpenSmile.

optional arguments:
  -h, --help            show this help message and exit
  --audio AUDIO         path to the input audio file, such as '/path/to/audio.wav'
  --diarization DIARIZATION
                        path to the input diarization file, such as '/path/to/diarization.rttm'
  --features FEATURES   path to the output feature archive, such as '/path/to/opensmile.tar.gz'
  --feature_set FEATURE_SET
                        available sets: ['ComParE_2016', 'GeMAPSv01a', 'GeMAPSv01b', 'eGeMAPSv01a', 'eGeMAPSv01b', 'eGeMAPSv02', 'emobase']
  --feature_level FEATURE_LEVEL
                        available levels: ['lld', 'lld_de', 'func']
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

### Video

#### Manipulation

##### Process

```
psifx video manipulation process [-h] --in_video IN_VIDEO --out_video
                                        OUT_VIDEO [--start START] [--end END]
                                        [--x_min X_MIN] [--y_min Y_MIN]
                                        [--x_max X_MAX] [--y_max Y_MAX]
                                        [--width WIDTH] [--height HEIGHT]
                                        [--overwrite | --no-overwrite]
                                        [--verbose | --no-verbose]

    Tool for processing videos.
    The trimming, cropping and resizing can be performed all at once, and in that order.

optional arguments:
  -h, --help            show this help message and exit
  --in_video IN_VIDEO   path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --out_video OUT_VIDEO
                        path to the output video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --start START         trim: timestamp in seconds of the start of the selection
  --end END             trim: timestamp in seconds of the end of the selection
  --x_min X_MIN         crop: x-axis coordinate of the top-left corner in pixels
  --y_min Y_MIN         crop: y-axis coordinate of the top-left corner in pixels
  --x_max X_MAX         crop: x-axis coordinate of the bottom-right corner in pixels
  --y_max Y_MAX         crop: y-axis coordinate of the bottom-right corner in pixels
  --width WIDTH         resize: width of the resized output
  --height HEIGHT       resize: height of the resized output
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

#### Pose Estimation

##### Inference

```
psifx video pose mediapipe inference [-h] --video VIDEO --poses POSES
                                            [--masks MASKS]
                                            [--mask_threshold MASK_THRESHOLD]
                                            [--model_complexity MODEL_COMPLEXITY]
                                            [--smooth | --no-smooth]
                                            [--device DEVICE]
                                            [--overwrite | --no-overwrite]
                                            [--verbose | --no-verbose]

    Tool for inferring human pose with MediaPipe Holistic.

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --poses POSES         path to the output pose archive, such as '/path/to/poses.tar.gz'
  --masks MASKS         path to the output segmentation mask video file, such as '/path/to/masks.mp4' (or .avi, .mkv, etc.)
  --mask_threshold MASK_THRESHOLD
                        threshold for the binarization of the segmentation mask
  --model_complexity MODEL_COMPLEXITY
                        complexity of the model: {0, 1, 2}, higher means more FLOPs, but also more accurate results
  --smooth, --no-smooth
                        temporally smooth the inference results to reduce the jitter (default: True)
  --device DEVICE       device on which to run the inference, either 'cpu' or 'cuda'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Visualization

```
psifx video pose mediapipe visualization [-h] --video VIDEO --poses
                                                POSES --visualization
                                                VISUALIZATION
                                                [--confidence_threshold CONFIDENCE_THRESHOLD]
                                                [--overwrite | --no-overwrite]
                                                [--verbose | --no-verbose]

    Tool for visualizing the poses over the video.

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --poses POSES         path to the input pose archive, such as '/path/to/poses.tar.gz'
  --visualization VISUALIZATION
                        path to the output visualization video file, such as '/path/to/visualization.mp4' (or .avi, .mkv, etc.)
  --confidence_threshold CONFIDENCE_THRESHOLD
                        threshold for not displaying low confidence keypoints
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

#### Face Analysis

##### Inference

```
psifx video face openface inference [-h] --video VIDEO --features
                                           FEATURES
                                           [--overwrite | --no-overwrite]
                                           [--verbose | --no-verbose]

    Tool for inferring face features from videos with OpenFace.

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --features FEATURES   path to the output feature archive, such as '/path/to/openface.tar.gz'
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

##### Visualization

```
psifx video face openface visualization [-h] --video VIDEO --features
                                               FEATURES --visualization
                                               VISUALIZATION [--depth DEPTH]
                                               [--f_x F_X] [--f_y F_Y]
                                               [--c_x C_X] [--c_y C_Y]
                                               [--overwrite | --no-overwrite]
                                               [--verbose | --no-verbose]

    Tool for visualizing face features from videos with OpenFace.

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         path to the input video file, such as '/path/to/video.mp4' (or .avi, .mkv, etc.)
  --features FEATURES   path to the input feature archive, such as '/path/to/openface.tar.gz'
  --visualization VISUALIZATION
                        path to the output video file, such as '/path/to/visualization.mp4' (or .avi, .mkv, etc.)
  --depth DEPTH         projection: assumed static depth of the subject in meters
  --f_x F_X             projection: x-axis of the focal length
  --f_y F_Y             projection: y-axis of the focal length
  --c_x C_X             projection: x-axis of the principal point
  --c_y C_Y             projection: y-axis of the principal point
  --overwrite, --no-overwrite
                        overwrite existing files, otherwise raises an error (default: False)
  --verbose, --no-verbose
                        verbosity of the script (default: True)
```

## Examples

## Development

### Variables

```bash
export PSIFX_VERSION="0.0.2"
export HF_TOKEN="write-your-hf-token-here"

docker buildx build \
  --build-arg PSIFX_VERSION=$PSIFX_VERSION \
  --build-arg HF_TOKEN=$HF_TOKEN \
  --tag "psifx:$VERSION" \
  --push .
```
