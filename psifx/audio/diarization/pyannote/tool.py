from typing import Union, Dict, Optional

from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from psifx.audio.diarization.tool import DiarizationTool
from psifx.utils.text_writer import RTTMWriter


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

        self.writer = RTTMWriter()

    def inference(
        self,
        audio_path: Union[str, Path],
        diarization_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(diarization_path, Path):
            diarization_path = Path(diarization_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"diarization     =   {diarization_path}")

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        # diarization_results is a pyannote.core.Annotation
        diarization_results: Annotation = self.model(
            file=audio_path,
            num_speakers=num_speakers,
        )

        # POST-PROCESSING
        # Converting it to a simple and practical dictionary.
        diarization_results: Dict = {
            "segments": [
                {
                    "uri": audio_path.stem,
                    "start": segment.start,
                    "end": segment.end,
                    "label": label,
                }
                for segment, track, label in diarization_results.itertracks(
                    yield_label=True
                )
            ]
        }

        if diarization_path.exists():
            if self.overwrite:
                diarization_path.unlink()
            else:
                raise FileExistsError(diarization_path)
        diarization_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer(
            result=diarization_results,
            path=diarization_path,
        )


def inference_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the input audio or directory containing the input audios.",
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to the output diarization or directory containing the diarizations.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Number of speaking participants, if ignored the model will try to guess it, it is advised to specify it. "
        "If a directory is passed as input, the number of speakers will be the same for all the audio tracks.",
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
