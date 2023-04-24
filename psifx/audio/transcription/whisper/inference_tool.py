from typing import Union, Dict, Optional

from pathlib import Path
import json

import torch
import torchaudio

from whisper import Whisper, load_model

from psifx.audio.transcription.tool import TranscriptionTool
from psifx.utils.text_writer import (
    TXTWriter,
    VTTWriter,
    SRTWriter,
    TSVWriter,
    JSONWriter,
)


WRITERS = {
    w.suffix: w for w in [TXTWriter, VTTWriter, SRTWriter, TSVWriter, JSONWriter]
}


class WhisperTranscriptionTool(TranscriptionTool):
    def __init__(
        self,
        model_name: str = "small",
        transcription_suffix: str = ".vtt",
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
        self.model: Whisper = load_model(model_name, device=self.device)
        # Freeze the model.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.writer = WRITERS[transcription_suffix]()

    def inference(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
        language: Optional[str] = None,
    ):
        if not isinstance(audio_path, Path):
            audio_path = Path(audio_path)
        if not isinstance(transcription_path, Path):
            transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        transcription_results: Dict = self.model.transcribe(
            audio=str(audio_path),
            task="transcribe",
            language=language,
            verbose=self.verbose > 1,
        )

        # POST-PROCESSING
        # Nothing to do here, it is already formatted.

        if transcription_path.exists():
            if self.overwrite:
                transcription_path.unlink()
            else:
                raise FileExistsError(transcription_path)
        transcription_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer(
            result=transcription_results,
            path=transcription_path,
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
        default=None,
        help="Path to the input diarization or directory containing the input diarizations.",
    )
    parser.add_argument(
        "--speaker_assignment",
        type=Path,
        default=None,
        help="Path to the input speaker assignment or directory containing the input speaker assignments.",
    )
    parser.add_argument(
        "--transcription",
        type=Path,
        required=True,
        help="Path to the output transcription file or directory containing the transcriptions.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="VTT",
        help="Format of the transcription. Available formats: TXT, VTT, SRT, TSV, JSON",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of the audio, if ignore, the model will try to guess it, it is advised to specify it. "
        "If a directory is passed as input, the language will be the same for all the audio tracks.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="small",
        help="Name of the model, check https://github.com/openai/whisper#available-models-and-languages.",
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

    tool = WhisperTranscriptionTool(
        model_name=args.model_name,
        transcription_suffix=f".{args.format.lower()}",
        device=args.device,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    tool.inference(
        audio_path=args.audio,
        transcription_path=args.transcription,
        language=args.language,
    )
    del tool
