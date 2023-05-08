from typing import Union, Sequence

from pathlib import Path

from psifx.tool import BaseTool


class IdentificationTool(BaseTool):
    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        mono_audio_paths: Sequence[Union[str, Path]],
        identification_path: Union[str, Path],
    ):
        raise NotImplementedError
