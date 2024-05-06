"""face analysis command-line interface."""

import argparse

from psifx.utils.command import Command, register_command
from psifx.video.face.openface.command import OpenFaceCommand


class FaceAnalysisCommand(Command):
    """
    Command-line interface for estimating face features from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "openface", OpenFaceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
