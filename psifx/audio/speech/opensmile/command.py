import argparse
from pathlib import Path

from psifx.audio.speech.opensmile.tool import (
    FEATURE_SETS,
    FEATURE_LEVELS,
    OpenSmileSpeechTool,
)
from psifx.command import Command, register_command


class OpenSmileCommand(Command):
    """
    Tool for running OpenSmile.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", OpenSmileInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class OpenSmileInferenceCommand(Command):
    """
    Tool for extracting non-verbal speech features from an audio track with OpenSmile.
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
            "--diarization",
            type=Path,
            required=True,
            help="Path to the diarization file.",
        )
        parser.add_argument(
            "--features",
            type=Path,
            required=True,
            help="Path to the features file.",
        )
        parser.add_argument(
            "--feature_set",
            type=str,
            default="ComParE_2016",
            help=f"Available sets: {list(FEATURE_SETS.keys())}",
        )
        parser.add_argument(
            "--feature_level",
            type=str,
            default="func",
            help=f"Available levels: {list(FEATURE_LEVELS.keys())}",
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
