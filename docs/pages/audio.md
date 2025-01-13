# Audio Processing Guide

This document covers the `psifx` commands for audio pre-processing, inference, and visualization.

## 1. Audio Pre-processing

### Audio Extraction
Extracts audio from a video file.
```bash
psifx audio manipulation extraction \
    --video Videos/Left.mp4 \
    --audio Audios/Left.wav
```
- `--video`: Path to the input video file.
- `--audio`: Path for the output audio file.


### Audio Conversion:
Converts any audio track to a mono audio track at 16kHz sample rate.
```bash
psifx audio manipulation conversion \
    --audio Audios/Left.mp4 \
    --mono_audio Audios/MonoLeft.wav
```
- `--audio`: Path to the input audio file.
- `--mono_audio`: Path to the output audio file.


### Audio Mixing
Combines multiple mono audio files into one.
```bash
psifx audio manipulation mixdown \
    --mono_audios Audios/MonoRight.wav Audios/MonoLeft.wav \
    --mixed_audio Audios/Mixed.wav
```
- `--mono_audios`: Paths of input mono audio files to combine.
- `--mixed_audio`: Path for the output mixed audio file.

### Audio Normalization
Normalizes the volume level of an audio file.
```bash
psifx audio manipulation normalization \
    --audio Audios/Mixed.wav \
    --normalized_audio Audios/MixedNormalized.wav
```
- `--audio`: Path to the input audio file.
- `--normalized_audio`: Path for the output normalized audio file.

## 2. Inference

### Speaker Diarization
Identifies segments for each speaker in an audio file.
```bash
psifx audio diarization pyannote inference \
    --audio Audios/MixedNormalized.wav \
    --diarization Diarizations/Mixed.rttm \
    [--num_speakers 2] \
    [--device cuda] \
    [--api_token hf_SomeLetters] \
    [--model_name pyannote/speaker-diarization@2.1.1]
```
- `--audio`: Input audio file for diarization.
- `--diarization`: Path to save diarization results in `.rttm` format.
- `--num_speakers`: Number of speakers to identify.
- `--device`: Processing device (`cuda` for GPU, `cpu` for CPU), default `cpu`.
- `--api_token`: Hugging Face token, may be required to download the model. Can also be provided as the environment variable **HF_TOKEN**.
- `--model_name`: Name of the diarization model used, default `pyannote/speaker-diarization@2.1.1`.

### Speaker Identification
Associates speakers in a mixed audio file with known audio samples.
```bash
psifx audio identification pyannote inference \
    --mixed_audio Audios/MixedNormalized.wav \
    --diarization Diarizations/Mixed.rttm \
    --mono_audios Audios/LeftNormalized.wav Audios/RightNormalized.wav \
    --identification Identifications/Mixed.json \
    [--device cuda] \
    [--api_token hf_SomeLetters] \
    [--model_names pyannote/embedding speechbrain/spkrec-ecapa-voxceleb]
``` 
- `--mixed_audio`: Mixed mono audio file with multiple speakers.
- `--diarization`: Diarization results for speaker segmentation.
- `--mono_audios`: Paths of known mono audio samples for each speaker.
- `--identification`: Path for saving identification results in `.json` format.
- `--device`: Processing device (`cuda` for GPU, `cpu` for CPU), default `cpu`.
- `--api_token`: Hugging Face token, may be required to download the model. Can also be provided as the environment variable **HF_TOKEN**.
- `--model_names`: Names of the embedding models.


### Speech Transcription
Transcribes speech in an audio file to text.
```bash
psifx audio transcription whisper inference \
    --audio Audios/MixedNormalized.wav \
    --transcription Transcriptions/Mixed.vtt \
    [--model_name small] \
    [--language fr] \
    [--device cuda] \
    [--translate_to_english] 
```
- `--audio`: Input audio file for transcription.
- `--transcription`: Path to save the transcription in `.vtt` format.
- `--model_name`: Model name (e.g., `large` for higher accuracy) check [Whisper](https://github.com/openai/whisper#available-models-and-languages), default `small`.
- `--language`: Language of the audio content.
- `--device`: Processing device (`cuda` for GPU, `cpu` for CPU), default `cpu`.
- `--translate_to_english`: Whether to transcribe the audio in its original language or to translate it to english, default `False`.

### Enhanced Transcription
Enhances transcription with diarization and speaker labels.
```bash
psifx audio transcription whisper enhance \
    --transcription Transcriptions/Mixed.vtt \
    --diarization Diarizations/Mixed.rttm \
    --identification Identifications/Mixed.json \
    --enhanced_transcription Transcriptions/MixedEnhanced.vtt
```
- `--transcription`: Path to the initial transcription file.
- `--diarization`: Diarization data file.
- `--identification`: Identification data file.
- `--enhanced_transcription`: Path to save the enhanced transcription.

## 3. Visualization

### Diarization Visualization
Generates a visual timeline of speaker segments.
```bash
psifx audio diarization visualization \
    --diarization Diarizations/Mixed.rttm \
    --visualization Visualizations/Mixed.png
```
- `--diarization`: Diarization data file.
- `--visualization`: Path for saving the visualization image.