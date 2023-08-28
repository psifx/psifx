import argparse
from pathlib import Path

from psifx.audio.manipulation.tool import ManipulationTool
from psifx.command import Command, register_command


class ManipulationCommand(Command):
    """
    Tools for manipulating audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "extraction", ExtractionCommand)
        register_command(subparsers, "conversion", ConversionCommand)
        register_command(subparsers, "mixdown", MixDownCommand)
        register_command(subparsers, "normalization", NormalizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class ExtractionCommand(Command):
    """
    Tool for extracting the audio track from a video.
    Expected video input extension: path/to/video.{any ffmpeg-readable}
    Expected audio output extension: path/to/audio.wav
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
            help="path to the video",
        )
        parser.add_argument(
            "--audio",
            type=Path,
            required=True,
            help="path to the audio",
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
        tool = ManipulationTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.extraction(
            video_path=args.video,
            audio_path=args.audio,
        )
        del tool


class ConversionCommand(Command):
    """
    Tool for converting an audio track to a .wav audio track with 16kHz sample rate.
    Expected audio input extension: path/to/audio.{any ffmpeg-readable}
    Expected audio output extension: path/to/audio.wav
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
            "--mono_audio",
            type=Path,
            required=True,
            help="path to the mono audio",
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
        tool = ManipulationTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.convert(
            audio_path=args.audio,
            mono_audio_path=args.mono_audio,
        )
        del tool


class MixDownCommand(Command):
    """
    Tool for mixing multiple mono audio tracks.
    Expected mono audio input extension(s): path/to/mono-audio.wav
    Expected mixed audio output extension: path/to/mixed-audio.wav
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--mono_audios",
            nargs="+",
            type=Path,
            required=True,
            help="path to the mono audios",
        )
        parser.add_argument(
            "--mixed_audio",
            type=Path,
            required=True,
            help="path to the mixed audio",
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
        tool = ManipulationTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.mixdown(
            mono_audio_paths=args.mono_audios,
            mixed_audio_path=args.mixed_audio,
        )
        del tool


class NormalizationCommand(Command):
    """
    Tool for normalizing an audio track.
    Expected audio input extension: path/to/audio.wav
    Expected normalized audio output extension: path/to/normalized-audio.wav
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
            "--normalized_audio",
            type=Path,
            required=True,
            help="path to the normalized audio",
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
        tool = ManipulationTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.normalization(
            audio_path=args.audio,
            normalized_audio_path=args.normalized_audio,
        )
        del tool
