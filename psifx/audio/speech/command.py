"""speech processing command-line interface."""

import argparse

from psifx.utils.command import Command, register_command


class SpeechCommand(Command):
    """
    Command-line interface for extracting non-verbal speech features from an audio track.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.audio.speech.opensmile.command import OpenSmileCommand

        register_command(subparsers, "opensmile", OpenSmileCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
