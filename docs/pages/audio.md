# Audio Processing Guide

This document covers the `psifx` commands for audio pre-processing, inference, and visualization.

## 1. Audio Pre-processing

### Audio Extraction
Extracts audio from a video file.
```bash
psifx audio manipulation extraction \
    --video Videos/Video.mp4 \
    --audio Audios/Audio.wav
```
- `--video`: Path to the input video file.
- `--audio`: Path for the output audio file.


### Audio Conversion
Converts any audio track to a mono audio track at 16kHz sample rate.
```bash
psifx audio manipulation conversion \
    --audio Audios/Audio.wav \
    --mono_audio Audios/MonoAudio.wav
```
- `--audio`: Path to the input audio file.
- `--mono_audio`: Path to the output audio file.

### Audio Split
Split a stereo audio track into two mono audio tracks.
```bash
psifx audio manipulation split \
    --stereo_audio  Audios/Audio.wav \
    --left_audio Audios/MonoLeft.wav \
    --right_audio Audios/MonoRight.wav
```
- `--stereo-audio`: Path to the input stereo audio file.
- `--left_audio`: Path to the output left channel mono audio file.
- `--right_audio`: Path to the output right channel mono audio file.


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


### Audio Trimming
Trims the audio file to be within given start and end times (in seconds). Both are optional (e.g. in the absence of a start time, only the end time will be used to reduce the length).
```bash
psifx audio manipulation trim \
    --audio Audios/Mixed.wav \
    --trimmed_audio Audios/MixedTrimmed.wav \
    [--start_time 5] \
    [--end_time 25.8]
```


## 2. Inference

### Speaker Diarization
Identifies segments for each speaker in an audio file. This first function takes in a 'mixed' .wav file, which is a down-mix of all the individual channels from each available microphone. 

For best results, the audio should derive from collar-/shirt-worn lavalier microphones.

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
Associates speakers in a mixed audio file with known audio samples. This combines the mixdown file with the individual channels of audio, and performs re-embedding and clustering, using the names of the individual audio files to assign identities in the identification out json. The purpose is (a) to improve diarization, and (b) to provide a mapping from the allocated/detected speakers from in the diarization process to enhance the transcription with.

```bash
psifx audio identification pyannote inference \
    --mixed_audio Audios/MixedNormalized.wav \
    --diarization Diarizations/Mixed.rttm \
    --mono_audios Audios/MonoRight.wav Audios/MonoLeft.wav \
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

The output of the identification is a `.json` structured like this:

```json
{"mapping": {"SPEAKER_00": "patient_micropone.wav", "SPEAKER_01": "therapist_microphone.wav"}, "agreement": 0.7874015748031497}
```

So if necessary, this can be manually edited and used for subsequent transcription enhancement.

### Speech Transcription
Use Whisper to transcribe speech in an audio file to text. Two implementations are available: OpenAI Whisper and HuggingFace Whisper.

#### OpenAI Whisper
```bash
psifx audio transcription whisper openai inference \
    --audio Audios/MixedNormalized.wav \
    --transcription Transcriptions/Mixed.vtt \
    [--model_name small] \
    [--language fr] \
    [--device cuda] \
    [--translate_to_english] 
```
- `--audio`: Input audio file for transcription.
- `--transcription`: Path to save the transcription in `.vtt` format.
- `--model_name`: Name of the OpenAI Whisper model (default: `small`).  
  Available models: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`.  
  See [official models](https://github.com/openai/whisper#available-models-and-languages) for details.
- `--language`: Two-letter language code of the audio content (e.g., `en` for English, `fr` for French, `de` for German).  
  If not specified, the model will attempt to auto-detect the language, but this may be less accurate.  
  It is recommended to specify the language when known.
- `--device`: Processing device (`cuda` for GPU, `cpu` for CPU, default: `cpu`).
- `--translate_to_english`: Whether to transcribe the audio in its original language or translate it to English (default: `False`).

#### HuggingFace Whisper
```bash
psifx audio transcription whisper huggingface inference \
    --audio Audios/MixedNormalized.wav \
    --transcription Transcriptions/Mixed.vtt \
    [--model_name "openai/whisper-small"] \
    [--language fr] \
    [--device cuda] \
    [--translate_to_english] 
```
- `--audio`: Input audio file for transcription.
- `--transcription`: Path to save the transcription in `.vtt` format.
- `--model_name`: Name of the HuggingFace model (default: `openai/whisper-small`).  
  Can use any [Whisper model from HuggingFace](https://huggingface.co/models?other=whisper).  
  Example: `nizarmichaud/whisper-large-v3-turbo-swissgerman`
- `--language`: Two-letter language code of the audio content (e.g., `en` for English, `fr` for French, `de` for German).  
  If not specified, the model will attempt to auto-detect the language, but this may be less accurate.  
  It is recommended to specify the language when known.
- `--device`: Processing device (`cuda` for GPU, `cpu` for CPU, default: `cpu`).
- `--translate_to_english`: Whether to transcribe the audio in its original language or translate it to English (default: `False`).  

### Enhanced Transcription
Enhances the transcription using the speaker labels from the diarization process.
```bash
psifx audio transcription enhance \
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
