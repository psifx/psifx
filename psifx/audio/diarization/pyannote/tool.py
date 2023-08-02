from typing import Optional, Union

from pathlib import Path
from tqdm import tqdm

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from psifx.audio.diarization.tool import DiarizationTool
from psifx.io import rttm, wav


class PyannoteDiarizationTool(DiarizationTool):
    def __init__(
        self,
        model_name: str = "2.1.1",
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

        if api_token is None:
            api_token = "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"

        self.model_name = model_name
        self.api_token = api_token

        self.model: Pipeline = Pipeline.from_pretrained(
            checkpoint_path=f"pyannote/speaker-diarization@{model_name}",
            use_auth_token=api_token,
        ).to(device=torch.device(device))

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ):
        audio_path = Path(audio_path)
        diarization_path = Path(diarization_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")

        wav.WAVReader.check(audio_path)
        rttm.RTTMWriter.check(diarization_path)

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        for i in tqdm(
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
            path=diarization_path,
            segments=segments,
            overwrite=self.overwrite,
        )
