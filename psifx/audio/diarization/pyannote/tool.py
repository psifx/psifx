from typing import Optional, Union

from pathlib import Path
from tqdm import tqdm

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from psifx.audio.diarization.tool import DiarizationTool
from psifx.io import rttm


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
        ).to(device=device)

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


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the audio file.",
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to the diarization file.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Number of speaking participants, if ignored the model will try to guess it, it is advised to specify it. ",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="2.1.1",
        help="Version number of the pyannote/speaker-diarization model, c.f. https://huggingface.co/pyannote/speaker-diarization/tree/main/reproducible_research",
    )
    parser.add_argument(
        "--api_token",
        type=str,
        default=None,
        help="API token for the downloading the models from HuggingFace.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to run the inference, either 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files, otherwise raises an error.",
    )
    parser.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Verbosity of the script.",
    )
    args = parser.parse_args()

    tool = PyannoteDiarizationTool(
        model_name=args.model_name,
        api_token=args.api_token,
        device=args.device,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        audio_path=args.audio,
        diarization_path=args.diarization,
        num_speakers=args.num_speakers,
    )
    del tool
