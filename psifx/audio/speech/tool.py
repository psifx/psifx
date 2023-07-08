from typing import Union

from pathlib import Path

from psifx.tool import BaseTool


class SpeechTool(BaseTool):
    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        features_path: Union[str, Path],
    ):
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)
        features_path = Path(features_path)

        # audio = load(audio_path)
        # audio = pre_process_func(audio)
        # features = model(audio)
        # features = post_process_func(features)
        # write(features, features_path)

        raise NotImplementedError