conda activate psifx-env

cd /home/guillaume/Datasets/UNIL/230520_1701


psifx-video-pose-mediapipe-inference --video Videos/Right.mp4 --poses Poses/Right.tar.xz --masks Masks/Right.mp4 --model_complexity 2 --overwrite
psifx-video-pose-mediapipe-inference --video Videos/Left.mp4 --poses Poses/Left.tar.xz --masks Masks/Left.mp4 --model_complexity 2 --overwrite

psifx-video-face-openface-inference --video Videos/Right.mp4 --features Faces/Right.tar.xz --overwrite
psifx-video-face-openface-inference --video Videos/Left.mp4 --features Faces/Left.tar.xz --overwrite

psifx-audio-manipulation-extraction --video Videos/Right.mp4 --audio Audios/Right.wav --overwrite
psifx-audio-manipulation-extraction --video Videos/Left.mp4 --audio Audios/Left.wav --overwrite

psifx-audio-manipulation-mixdown --mono_audios Audios/Right.wav Audios/Left.wav --mixed_audio Audios/Mixed.wav --overwrite

psifx-audio-manipulation-normalization --audio Audios/Right.wav --normalized_audio Audios/Right.normalized.wav
psifx-audio-manipulation-normalization --audio Audios/Left.wav --normalized_audio Audios/Left.normalized.wav
psifx-audio-manipulation-normalization --audio Audios/Mixed.wav --normalized_audio Audios/Mixed.normalized.wav

psifx-audio-diarization-pyannote-inference --audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.normalized.rttm --num_speakers 2 --device cuda --overwrite

psifx-audio-transcription-whisper-inference --audio Audios/Mixed.normalized.wav --transcription Transcriptions/Mixed.normalized.vtt --model_name small --language fr --device cuda --overwrite

psifx-audio-identification-pyannote-inference --audio Audios/Mixed.normalized.wav --diarization Diarizations/Mixed.normalized.rttm --mono_audios Audios/Left.normalized.wav Audios/Right.normalized.wav --identification Identifications/Mixed.normalized.json --device cuda --overwrite