from psifx.audio.transcription.whisper import inference_tool


def main():
    tool = inference_tool.WhisperTranscriptionWithDiarizationTool(
        model_name="small",
        transcription_suffix=".vtt",
        device="cuda",
        overwrite=True,
        verbose=True,
    )

    tool(
        audio_path="/home/guillaume/Datasets/UNIL/CH.101/RawAudios/CH.101.f.wav",
        diarization_path="/home/guillaume/Datasets/UNIL/CH.101/Diarizations/CH.101.f.rttm",
        speaker_assignment_path="/home/guillaume/Datasets/UNIL/CH.101/SpeakerAssignment/CH.101.f.json",
        transcription_path="/home/guillaume/Datasets/UNIL/CH.101/Transcriptions/CH.101.f.vtt",
        language="fr",
    )


if __name__ == "__main__":
    main()
