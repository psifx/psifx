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
    Expected audio input extension: path/to/audio.wav
    Expected diarization input extension: path/to/diarization.rttm
    Expected feature extraction output extension: path/to/opensmile-features.tar
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
            "--diarization",
            type=Path,
            required=True,
            help="path to the diarization",
        )
        parser.add_argument(
            "--features",
            type=Path,
            required=True,
            help="path to the feature archive",
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
