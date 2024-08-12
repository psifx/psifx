# Audio

## Pre-processing

```bash
psifx audio manipulation extraction --video Videos/Left.mp4 --audio Audios/Left.wav
psifx audio manipulation extraction --video Videos/Right.mp4 --audio Audios/Right.wav

psifx audio manipulation mixdown --mono_audios Audios/Right.wav Audios/Left.wav --mixed_audio Audios/Mixed.wav

psifx audio manipulation normalization --audio Audios/Right.wav --normalized_audio Audios/Right.normalized.wav
psifx audio manipulation normalization --audio Audios/Left.wav --normalized_audio Audios/Left.normalized.wav
psifx audio manipulation normalization --audio Audios/Mixed.wav --normalized_audio Audios/Mixed.normalized.wav
```

## Inference

```bash
psifx audio diarization pyannote inference --audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.rttm --num_speakers 2 --device cuda

psifx audio identification pyannote inference --mixed_audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.rttm --mono_audios Audios/Left.normalized.wav Audios/Right.normalized.wav --identification Identifications/Mixed.json --device cuda

psifx audio transcription whisper inference --audio Audios/Mixed.normalized.wav --transcription Transcriptions/Mixed.vtt --model_name large --language fr --device cuda

psifx audio transcription whisper enhance --transcription Transcriptions/Mixed.vtt --diarization Diarizations/Mixed.rttm --identification Identifications/Mixed.json --enhanced_transcription Transcriptions/Mixed.enhanced.vtt
```

## Visualization

```bash
psifx audio diarization visualization --diarization Diarizations/Mixed.rttm --visualization Visualizations/Mixed.png
```
