"""Whisper transcription command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.tool import WhisperTranscriptionTool, HuggingFaceTranscriptionTool


class WhisperCommand(Command):
    """
    Command-line interface for running Whisper.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        from psifx.audio.transcription.command import EnhancedTranscriptionCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", WhisperTranscriptionCommand)
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


class WhisperTranscriptionCommand(Command):
    """
    Command-line interface for transcribing an audio track with Whisper.
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
            help="language of the audio, if ignore, the model will try to guess it, "
                 "it is advised to specify it",
        )
        parser.add_argument(
            "--use_hf",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="use model from hugging face (by default use openai-whisper)",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="name of the model, check https://github.com/openai/whisper#available-models-and-languages; for hf check https://huggingface.co/models?other=whisper instead",
        )

        parser.add_argument(
            "--translate_to_english",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="whether to transcribe the audio in its original language or"
                 " to translate it to english",
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
        tool_class = HuggingFaceTranscriptionTool if args.use_hf else WhisperTranscriptionTool

        tool_kwargs = {
            "device": args.device,
            "overwrite": args.overwrite,
            "verbose": args.verbose,
        }
        if args.model_name is not None:
            tool_kwargs["model_name"] = args.model_name

        tool = tool_class(**tool_kwargs)

        tool.inference(
            audio_path=args.audio,
            transcription_path=args.transcription,
            task="transcribe" if not args.translate_to_english else "translate",
            language=args.language,
        )
        del tool
