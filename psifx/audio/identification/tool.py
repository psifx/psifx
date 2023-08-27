from typing import Union, Sequence

from pathlib import Path

from psifx.tool import BaseTool


class IdentificationTool(BaseTool):
    """
    Base class for identification tools.
    """

    def inference(
        self,
        mixed_audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        mono_audio_paths: Sequence[Union[str, Path]],
        identification_path: Union[str, Path],
    ):
        """
        Skeleton of the inference method.

        :param mixed_audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :param mono_audio_paths:Path to the mono audio tracks.
        :param identification_path: Path to the identification file.
        :return:
        """
        raise NotImplementedError
