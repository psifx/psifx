import argparse
from pathlib import Path

from psifx.audio.identification.pyannote.tool import PyannoteIdentificationTool
from psifx.command import Command, register_command


class PyannoteCommand(Command):
    """
    Tool for running Pyannote.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", PyannoteInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class PyannoteInferenceCommand(Command):
    """
    Tool for identifying speakers from an audio track with Pyannote.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--mixed_audio",
            type=Path,
            required=True,
            help="path to the mixed audio",
        )
        parser.add_argument(
            "--diarization",
            type=Path,
            required=True,
            help="path to the diarization",
        )
        parser.add_argument(
            "--mono_audios",
            nargs="+",
            type=Path,
            required=True,
            help="paths to the mono audios",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="path to the identification",
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
