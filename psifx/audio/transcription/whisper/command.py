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
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "openai", OpenAIWhisperCommand)
        register_command(subparsers, "huggingface", HuggingFaceWhisperCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
