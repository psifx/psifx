import argparse
from pathlib import Path

from psifx.audio.transcription.whisper.tool import WhisperTranscriptionTool
from psifx.command import Command, register_command


class WhisperCommand(Command):
    """
    Tool for running Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.audio.transcription.command import EnhancedTranscriptionCommand

        register_command(subparsers, "inference", WhisperTranscriptionCommand)
        register_command(subparsers, "enhance", EnhancedTranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class WhisperTranscriptionCommand(Command):
    """
    Tool for transcribing an audio track with Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
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
