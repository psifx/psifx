from typing import Union

from pathlib import Path

from psifx.base_tool import BaseTool


class BaseDiarizationTool(BaseTool):
    def __call__(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(diarization_path, Path):
            diarization_path = Path(diarization_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # diarization = model(audio)
        # diarization = post_process_func(diarization)
        # write(diarization, diarization_path)

        raise NotImplementedError
