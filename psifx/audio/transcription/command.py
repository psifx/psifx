"""transcription command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.command import WhisperCommand
from psifx.audio.transcription.tool import TranscriptionTool


class TranscriptionCommand(Command):
    """
    Command-line interface for transcribing audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "whisper", WhisperCommand)
        register_command(subparsers, "enhance", EnhancedTranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class EnhancedTranscriptionCommand(Command):
    """
    Command-line interface for enhancing a transcription with diarization and identification.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--transcription",
            type=Path,
            required=True,
            help="path to the input transcription file, such as ``/path/to/transcription.vtt``",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the input diarization file, such as ``/path/to/diarization.rttm``",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="path to the input identification file, such as ``/path/to/identification.json``",
        )
        parser.add_argument(
            "--enhanced_transcription",
            type=Path,
            required=True,
            help="path to the output transcription file, such as ``/path/to/enhanced-transcription.vtt``",
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
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
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
