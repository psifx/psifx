# Psifx: Hands-On Example  

This document presents a detailed hands-on example showcasing the capabilities of **psifx**.  

## Objective  
The goal of this example is to process a video through the complete **psifx** pipeline, extracting as much information as possible. The video originates from a real experimental setting featuring a staged discussion between two individuals.  

## Process Overview  
We will apply **psifx** to:  
1. Extract **poses** and **facial features** from the video.  
2. **Diarize** and **identify** the audio to obtain a full transcript.  
3. Use an **LLM** to summarize the discussion based on the transcript.  

## Setup  
Let's begin by installing up psifx and getting the example video from the git repository.

<details>
  <summary> DOCKER </summary>

  
1. Install [Docker Engine](https://docs.docker.com/engine/install/#server) and make sure to follow
   the [post-install instructions](https://docs.docker.com/engine/install/linux-postinstall/). Otherwise,
   install [Docker Desktop](https://docs.docker.com/desktop/).
2. If you have a GPU and want to use it to accelerate compute:
    1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
    2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).

3. Clone the repo and navigate to the folder containing the example video.
   ```bash 
   git clone https://github.com/psifx/psifx.git
   cd psifx/example/data
   ```

4. Run the latest image with the example directory mounted.
   ```bash
   export DATA_PATH="$(pwd)"
   
   docker run \
      --user $(id -u):$(id -g) \
      --gpus all \
      --mount type=bind,source=$DATA_PATH,target=$DATA_PATH \
      --workdir $DATA_PATH \
      --interactive \
      --tty \
      psifx/psifx:latest
   ```


</details>

<details>
  <summary> LINUX </summary>

1. For Linux users, install the following system-wide:
   ```bash
   sudo apt install ffmpeg ubuntu-restricted-extras \
      build-essential cmake wget \
      libopenblas-dev \
      libopencv-dev \
      libdlib-dev \
      libboost-all-dev \
      libsqlite3-dev
   ```
2. Create a dedicated `conda` environment:
   ```bash
   conda create -y -n psifx-env python=3.11 pip
   conda activate psifx-env
   ```
3. Install `psifx`:
   ```bash
   pip install 'psifx @ git+https://github.com/psifx/psifx.git'
   ```
4. Verify your installation with:
   ```bash
   psifx
   ```

5. Install OpenFace using our fork:
   ```bash
   wget https://raw.githubusercontent.com/GuillaumeRochette/OpenFace/master/install.py && \
   python install.py && \
   rm install.py
   ```

6. Install [Ollama](https://github.com/ollama/ollama) locally.
   For Linux users, use this command:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

7. Clone the repo and navigate to the folder containing the example video.
   
   ```bash 
   git clone https://github.com/psifx/psifx.git
   cd psifx/example/data
   ```

</details>

## Process video

The video setting feature a staged discussion between two individuals.
To simplify the workflow only one point of view is provided.


### Tracking
To track and segment the individual, and to produce the associated segmentation mask.
```bash
psifx video tracking samurai inference \
    --video Video.mp4 \
    --mask_dir masks --overwrite
```
To visualize the tracking / segmentation masks:

```bash
psifx video tracking visualization \
    --video Video.mp4 \
    --masks masks \
    --visualization VisualizationTracking.mp4 --overwrite
```

### Pose
Detect and analyze human poses using MediaPipe.
```bash
psifx video pose mediapipe single-inference \
    --video Video.mp4 \
    --poses Poses.tar.gz --overwrite
```
Create a visual overlay of the poses detected in the video.
```bash
psifx video pose mediapipe visualization \
    --video Video.mp4 \
    --poses Poses.tar.gz \
    --visualization VisualizationPoses.mp4 --overwrite
```
### Face
Extract facial features from the video using OpenFace.
```bash
psifx video face openface single-inference \
    --video Video.mp4 \
    --features Faces.tar.gz --overwrite
```

Create a visual overlay of facial features detected in the video.
Add it on top of the overlay for poses.
```bash
psifx video face openface visualization \
    --video VisualizationPoses.mp4 \
    --features Faces.tar.gz \
    --visualization VisualizationFaces.mp4
```

## Process audio

For the purpose of the example, the video has stereo audio, where the left and right audio channels are fed uniquely from the two lavalier microphones for each person.

As such, the full 'audio scene' is embedded in the single video, and we can demonstrate the full pipeline for `psifx`.


### Preprocess audio
Extract the stereo audio track from the video (which contains left and right lavalier microphone outputs for this example).
```bash
psifx audio manipulation extraction \
    --video Video.mp4 \
    --audio Audio.wav
```
Recover the audio from each microphone by splitting the right and left channels.

```bash
psifx audio manipulation split \
    --stereo_audio  Audio.wav \
    --left_audio LeftAudio.wav \
    --right_audio RightAudio.wav
```

Convert the stereo audio to a mono audio track.
```bash
psifx audio manipulation conversion \
    --audio Audio.wav \
    --mono_audio Audio.wav \
    --overwrite
```

Normalize the volume level of each audio file.
```bash
psifx audio manipulation normalization \
    --audio Audio.wav \
    --normalized_audio Audio.wav \
    --overwrite
psifx audio manipulation normalization \
    --audio LeftAudio.wav \
    --normalized_audio LeftAudio.wav \
    --overwrite
psifx audio manipulation normalization \
    --audio RightAudio.wav \
    --normalized_audio RightAudio.wav \
    --overwrite
```


### Speaker Diarization
Identifies segments for each speaker in the audio file.
```bash
psifx audio diarization pyannote inference \
    --audio Audio.wav \
    --diarization Diarization.rttm
```

### Diarization Visualization
Generate a visual timeline of speaker segments.
```bash
psifx audio diarization visualization \
    --diarization Diarization.rttm \
    --visualization VisualizationDiarization.png
```

### Speaker Identification
Associates speakers in a mixed audio file with known audio samples.
```bash
psifx audio identification pyannote inference \
    --mixed_audio Audio.wav \
    --diarization Diarization.rttm \
    --mono_audios RightAudio.wav LeftAudio.wav \
    --identification Identification.json
``` 

### Speech Transcription
Transcribe speech in the audio file to text.
```bash
psifx audio transcription whisperx inference \
    --audio Audio.wav \
    --transcription Transcription.vtt
```

### Enhanced Transcription
Enhance transcription with diarization and speaker labels.
```bash
psifx audio transcription enhance \
    --transcription Transcription.vtt \
    --diarization Diarization.rttm \
    --identification Identification.json \
    --enhanced_transcription TranscriptionEnhanced.vtt
```

## Process text
We will use a language model to analyse the content of the transcription.

First create a yaml file Instruction.yaml containing:
```yaml
prompt: |
    user: Here is the transcription of a recording: {text}
    What are they talking about?
```

Run the following command to generate the file.
```bash
cat <<EOF > Instruction.yaml
prompt: |
    user: Here is the transcription of a recording: {text}
    What are they talking about?
EOF
```

Use a language model to analyse the content of the transcription according to some instruction.

```bash
psifx text instruction \
    --instruction Instruction.yaml \
    --input TranscriptionEnhanced.vtt \
    --output TranscriptionAnalysis.txt 
```