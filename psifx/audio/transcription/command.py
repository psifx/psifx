import argparse
from pathlib import Path

from psifx.audio.transcription.tool import TranscriptionTool
from psifx.command import Command, register_command


class TranscriptionCommand(Command):
    """
    Tools for transcribing audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.audio.transcription.whisper.command import WhisperCommand

        register_command(subparsers, "whisper", WhisperCommand)
        register_command(subparsers, "enhance", EnhancedTranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class EnhancedTranscriptionCommand(Command):
    """
    Tool for enhancing a transcription with diarization and identification.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--transcription",
            type=Path,
            required=True,
            help="Path to the transcription file.",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="Path to the diarization file.",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="Path to the identification file.",
        )
        parser.add_argument(
            "--enhanced_transcription",
            type=Path,
            required=True,
            help="Path to the enhanced transcription file.",
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
        tool = TranscriptionTool(
            device="cpu",
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.enhance(
            transcription_path=args.transcription,
            diarization_path=args.diarization,
            identification_path=args.identification,
            enhanced_transcription_path=args.enhanced_transcription,
        )
        del tool
