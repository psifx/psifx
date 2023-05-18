from typing import Union

from pathlib import Path

from psifx.tool import BaseTool


class TranscriptionTool(BaseTool):
    def inference(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
    ):
        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # transcription = model(audio)
        # transcription = post_process_func(transcription)
        # write(transcription, transcription_path)

        raise NotImplementedError

    def visualization(self, *args, **kwargs):
        print("Just open the .vtt file with a video player like VLC.")
        raise NotImplementedError

    def enhance(
        self,
        transcription_path: Union[str, Path],
        diarization_path: Union[str, Path],
        identification_path: Union[str, Path],
        enhanced_transcription_path: Union[str, Path],
    ):
        pass
