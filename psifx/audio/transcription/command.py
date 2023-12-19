import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.command import WhisperCommand
from psifx.audio.transcription.tool import TranscriptionTool


class TranscriptionCommand(Command):
    """
    Tools for transcribing audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "whisper", WhisperCommand)
        register_command(subparsers, "enhance", EnhancedTranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class EnhancedTranscriptionCommand(Command):
    """
    Tool for enhancing a transcription with diarization and identification.
    Expected transcription input extension: path/to/transcription.vtt
    Expected diarization input extension: path/to/diarization.rttm
    Expected identification input extension: path/to/identification.json
    Expected enhanced transcription output extension: path/to/enhanced-transcription.vtt
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--transcription",
            type=Path,
            required=True,
            help="path to the transcription",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the diarization",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="path to the identification",
        )
        parser.add_argument(
            "--enhanced_transcription",
            type=Path,
            required=True,
            help="path to the enhanced transcription",
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
