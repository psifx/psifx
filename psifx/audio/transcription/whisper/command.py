import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.tool import WhisperTranscriptionTool


class WhisperCommand(Command):
    """
    Tool for running Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        from psifx.audio.transcription.command import EnhancedTranscriptionCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", WhisperTranscriptionCommand)
        register_command(subparsers, "enhance", EnhancedTranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class WhisperTranscriptionCommand(Command):
    """
    Tool for transcribing an audio track with Whisper.
    Expected audio input extension: path/to/audio.wav
    Expected transcription output extension: path/to/transcription.vtt
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--audio",
            type=Path,
            required=True,
            help="path to the audio",
        )
        parser.add_argument(
            "--transcription",
            type=Path,
            required=True,
            help="path to the transcription",
        )
        parser.add_argument(
            "--language",
            type=str,
            default=None,
            help="language of the audio, if ignore, the model will try to guess it, "
            "it is advised to specify it",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="small",
            help="name of the model, check "
            "https://github.com/openai/whisper#available-models-and-languages",
        )
        parser.add_argument(
            "--translate_to_english",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="whether to transcribe the audio in its original language or"
            " to translate it to english",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
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
