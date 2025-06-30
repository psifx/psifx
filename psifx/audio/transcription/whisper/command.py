"""WhisperX transcription command-line interface."""
import argparse
from pathlib import Path

import torch

from psifx.audio.transcription.whisper.tool import WhisperXTool
from psifx.utils.command import Command, register_command


class WhisperXCommand(Command):
    """
    Command-line interface for running OpenAI Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", WhisperXInferenceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class WhisperXInferenceCommand(Command):
    """
    Command-line interface for transcribing an audio track with WhisperX.
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
            "--transcription",
            type=Path,
            required=True,
            help="path to the output transcription file, such as ``/path/to/transcription.vtt``",
        )
        parser.add_argument(
            "--language",
            type=str,
            default=None,
            help="language of the audio, if ignore, the model will try to guess it, it is advised to specify it",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="distil-large-v3",
            help="size of the model to use (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, "
                 "medium.en, distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3, "
                 "large-v3-turbo, or turbo), a path to a converted model directory, "
                 "or a CTranslate2-converted Whisper model ID from the HF Hub",
        )
        parser.add_argument(
            "--translate_to_english",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="whether to transcribe the audio in its original language or to translate it to english",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="batch size, reduce if low on GPU memory",
        )
        parser.add_argument(
            "--device",
            type=str,
            default= "cuda" if torch.cuda.is_available() else "cpu",
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
        tool = WhisperXTool(
            model_name=args.model_name,
            task="transcribe" if not args.translate_to_english else "translate",
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        tool.inference(
            audio_path=args.audio,
            transcription_path=args.transcription,
            batch_size=args.batch_size,
            language=args.language,
        )
        del tool
