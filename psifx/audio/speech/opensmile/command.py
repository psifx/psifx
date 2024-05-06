"""openSMILE speech processing command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.speech.opensmile.tool import (
    FEATURE_SETS,
    FEATURE_LEVELS,
    OpenSmileSpeechTool,
)


class OpenSmileCommand(Command):
    """
    Command-line interface for running OpenSmile.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", OpenSmileInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class OpenSmileInferenceCommand(Command):
    """
    Command-line interface for extracting non-verbal speech features from an audio track with OpenSmile.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--audio",
            type=Path,
            required=True,
            help="path to the input audio file, such as ``/path/to/audio.wav``",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the input diarization file, such as ``/path/to/diarization.rttm``",
        )
        parser.add_argument(
            "--features",
            type=Path,
            required=True,
            help="path to the output feature archive, such as ``/path/to/opensmile.tar.gz``",
        )
        parser.add_argument(
            "--feature_set",
            type=str,
            default="ComParE_2016",
            help=f"available sets: {list(FEATURE_SETS.keys())}",
        )
        parser.add_argument(
            "--feature_level",
            type=str,
            default="func",
            help=f"available levels: {list(FEATURE_LEVELS.keys())}",
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
        tool = OpenSmileSpeechTool(
            feature_set=args.feature_set,
            feature_level=args.feature_level,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            audio_path=args.audio,
            diarization_path=args.diarization,
            features_path=args.features,
        )
        del tool
