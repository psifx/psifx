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
            "--mono_audios",
            nargs="+",
            type=Path,
            required=True,
            help="Paths to the mono audio files.",
        )
        parser.add_argument(
            "--identification",
            type=Path,
            required=True,
            help="Path to the identification file.",
        )
        parser.add_argument(
            "--model_names",
            nargs="+",
            type=str,
            default=[
                "pyannote/embedding",
                "speechbrain/spkrec-ecapa-voxceleb",
            ],
            help="Version number of the pyannote/speaker-diarization model, c.f. https://huggingface.co/pyannote/speaker-diarization/tree/main/reproducible_research",
        )  # TODO
        parser.add_argument(
            "--api_token",
            type=str,
            default=None,
            help="API token for the downloading the models from HuggingFace.",
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
        tool = PyannoteIdentificationTool(
            model_names=args.model_names,
            api_token=args.api_token,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            mixed_audio_path=args.audio,
            diarization_path=args.diarization,
            mono_audio_paths=args.mono_audios,
            identification_path=args.identification,
        )
        del tool
