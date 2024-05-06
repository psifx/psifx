"""speaker identification command-line interface."""

import argparse

from psifx.utils.command import Command, register_command
from psifx.audio.identification.pyannote.command import PyannoteCommand


class IdentificationCommand(Command):
    """
    Command-line interface for identifying speakers in audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "pyannote", PyannoteCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
