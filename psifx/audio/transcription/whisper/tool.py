from typing import Union, Dict, Optional

from pathlib import Path

from whisper import Whisper, load_model

from psifx.audio.transcription.tool import TranscriptionTool
from psifx.io import vtt


class WhisperTranscriptionTool(TranscriptionTool):
    def __init__(
        self,
        model_name: str = "small",
        task: str = "transcribe",
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
        self.task = task
        self.model: Whisper = load_model(model_name, device=self.device)
        # Freeze the model.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def inference(
        self,
        audio_path: Union[str, Path],
        transcription_path: Union[str, Path],
        language: Optional[str] = None,
    ):
        audio_path = Path(audio_path)
        transcription_path = Path(transcription_path)

        if self.verbose:
            print(f"audio           =   {audio_path}")
            print(f"transcription   =   {transcription_path}")

        # PRE-PROCESSING
        # Nothing to do here, the model wants the path of the audio.

        # INFERENCE
        segments = self.model.transcribe(
            audio=str(audio_path),
            task=self.task,
            language=language,
            verbose=self.verbose > 1,
        )["segments"]

        # POST-PROCESSING
        vtt.VTTWriter.write(
            segments=segments,
            path=transcription_path,
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
        "--transcription",
        type=Path,
        required=True,
        help="Path to the transcription file.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of the audio, if ignore, the model will try to guess it, it is advised to specify it.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="small",
        help="Name of the model, check https://github.com/openai/whisper#available-models-and-languages.",
    )
    parser.add_argument(
        "--translate_to_english",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to transcribe the audio in its original language or to translate it to english.",
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
        task="transcribe" if not args.translate_to_english else "translate",
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
