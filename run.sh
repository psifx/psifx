cd Projects/psifx/
conda activate psifx-env
export DATA_ROOT=/home/guillaume/Datasets/UNIL/CH.102

psifx-video-pose-mediapipe-inference --video $DATA_ROOT/Videos/CH.102.R.mp4 --poses $DATA_ROOT/Poses/CH.102.R.tar.xz --masks $DATA_ROOT/Masks/CH.102.R.mp4 --model_complexity 2 --overwrite
psifx-video-pose-mediapipe-inference --video $DATA_ROOT/Videos/CH.102.L.mp4 --poses $DATA_ROOT/Poses/CH.102.L.tar.xz --masks $DATA_ROOT/Masks/CH.102.L.mp4 --model_complexity 2 --overwrite

psifx-video-face-openface-inference --video $DATA_ROOT/Videos/CH.102.R.mp4 --features $DATA_ROOT/Faces/CH.102.R.tar.xz --overwrite
psifx-video-face-openface-inference --video $DATA_ROOT/Videos/CH.102.L.mp4 --features $DATA_ROOT/Faces/CH.102.L.tar.xz --overwrite

psifx-audio-diarization-pyannote-inference --audio $DATA_ROOT/Audios/CH.102.combined.wav --diarization $DATA_ROOT/Diarizations/CH.102.combined.rttm --num_speakers 2 --device cuda --overwrite

psifx-audio-transcription-whisper-inference --audio $DATA_ROOT/Audios/CH.102.combined.wav --transcription $DATA_ROOT/Transcriptions/CH.102.combined.vtt --language fr --device cuda --overwrite

psifx-audio-identification-pyannote-inference --audio $DATA_ROOT/Audios/CH.102.combined.wav --diarization $DATA_ROOT/Diarizations/CH.102.combined.rttm --mono_audios $DATA_ROOT/Audios/CH.102.L.wav $DATA_ROOT/Audios/CH.102.R.wav --identification $DATA_ROOT/Identifications/CH.102.combined.json --device cuda --overwrite