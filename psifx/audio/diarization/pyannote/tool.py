"""pyannote speaker diarization tool."""

from typing import Optional, Union

from pathlib import Path
from tqdm import tqdm

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from psifx.audio.diarization.tool import DiarizationTool
from psifx.io import rttm, wav


class PyannoteDiarizationTool(DiarizationTool):
    """
    pyannote diarization tool.

    :param model_name: The name of the model to use.
    :param api_token: The HuggingFace API token to use.
    :param device: The device where the computation should be executed.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        api_token: Optional[str] = None,
        device: str = "cpu",
        overwrite: bool = False,
        verbose: Union[bool, int] = True,
    ):
        super().__init__(
            device=device,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.model_name = model_name
        self.api_token = api_token

        self.model: Pipeline = Pipeline.from_pretrained(
            checkpoint_path=model_name,
            use_auth_token=api_token,
        ).to(device=torch.device(device))

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ):
        """
        Implementation of pyannote's diarization inference method.

        :param audio_path: Path to the audio track.
        :param diarization_path: Path to the diarization file.
        :param num_speakers: Number of speaking participants, if ignored the model will try to guess it, it is advised to specify it.
        :return:
        """
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")

        wav.WAVReader.check(path=audio_path)
        rttm.RTTMWriter.check(path=diarization_path, overwrite=self.overwrite)

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        for _ in tqdm(
            range(1),
            desc="Processing",
            disable=not self.verbose,
        ):
            # results is a pyannote.core.Annotation
            results: Annotation = self.model(
                file=audio_path,
                num_speakers=num_speakers,
            )

        segments = [
            {
                "type": "SPEAKER",
                "file_stem": audio_path.stem,
                "channel": 1,
                "start": segment.start,
                "duration": segment.duration,
                "orthography": "<NA>",
                "speaker_type": "<NA>",
                "speaker_name": speaker_name,
                "confidence_score": "<NA>",
                "signal_lookahead_time": "<NA>",
            }
            for segment, track_name, speaker_name in tqdm(
                results.itertracks(yield_label=True),
                desc="Parsing",
                disable=not self.verbose,
            )
        ]
        rttm.RTTMWriter.write(
            segments=segments, path=diarization_path, overwrite=self.overwrite
        )
