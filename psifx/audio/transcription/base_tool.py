from typing import Union

from pathlib import Path

from psifx.base_tool import BaseTool


class BaseTranscriptionTool(BaseTool):
    def __call__(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(transcription_path, Path):
            transcription_path = Path(transcription_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # transcription = model(audio)
        # transcription = post_process_func(transcription)
        # write(transcription, transcription_path)

        raise NotImplementedError
