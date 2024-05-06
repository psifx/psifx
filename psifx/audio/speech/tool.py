"""speech processing tool."""

from typing import Union

from pathlib import Path

from psifx.tool import Tool


class SpeechTool(Tool):
    """
    Base class for non-verbal speech processing tools.
    """

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        """
        Template of the inference method.

        :param audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :param features_path: Path to the feature file.
        :return:
        """
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)
        features_path = Path(features_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # features = model(audio)
        # features = post_process_func(features)
        # write(features, features_path)

        raise NotImplementedError
