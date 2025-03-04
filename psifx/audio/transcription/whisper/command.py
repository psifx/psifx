"""Whisper transcription command-line interface."""
import argparse

from psifx.utils.command import Command, register_command
from psifx.audio.transcription.whisper.openai.command import OpenAIWhisperCommand
from psifx.audio.transcription.whisper.huggingface.command import HuggingFaceWhisperCommand


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

        whisper_parser = subparsers.add_parser("inference")
        whisper_subparsers = whisper_parser.add_subparsers(title="whisper implementation")
        register_command(whisper_subparsers, "openai", OpenAIWhisperCommand)
        register_command(whisper_subparsers, "huggingface", HuggingFaceWhisperCommand)
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
