"""pyannote speaker identification command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.identification.pyannote.tool import PyannoteIdentificationTool


class PyannoteCommand(Command):
    """
    Command-line interface for running pyannote identification tool.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", PyannoteInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class PyannoteInferenceCommand(Command):
    """
    Command-line interface for identifying speakers from an audio track with pyannote.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--mixed_audio",
            type=Path,
            required=True,
            help="path to the input mixed audio file, such as ``/path/to/mixed-audio.wav``",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the input diarization file, such as ``/path/to/diarization.rttm``",
        )
        parser.add_argument(
            "--mono_audios",
            nargs="+",
            type=Path,
            required=True,
            help="paths to the input mono audio files, such as ``/path/to/mono-audio-1.wav /path/to/mono-audio-2.wav``",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="path to the output identification file, such as ``/path/to/identification.json``",
        )
        parser.add_argument(
            "--model_names",
            nargs="+",
            type=str,
            default=[
                "pyannote/embedding",
                "speechbrain/spkrec-ecapa-voxceleb",
            ],
            help="names of the embedding models",
        )
        parser.add_argument(
            "--api_token",
            type=str,
            default=None,
            help="API token for the downloading the models from HuggingFace",
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
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        tool = PyannoteIdentificationTool(
            model_names=args.model_names,
            api_token=args.api_token,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            mixed_audio_path=args.mixed_audio,
            diarization_path=args.diarization,
            mono_audio_paths=args.mono_audios,
            identification_path=args.identification,
        )
        del tool
