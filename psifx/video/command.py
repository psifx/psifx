"""video processing command-line interface."""

import argparse

from psifx.utils.command import Command, register_command
from psifx.video.pose.command import PoseEstimationCommand
from psifx.video.manipulation.command import ManipulationCommand
from psifx.video.face.command import FaceAnalysisCommand


class VideoCommand(Command):
    """
    Command-line interface for processing videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "manipulation", ManipulationCommand)
        register_command(subparsers, "pose", PoseEstimationCommand)
        register_command(subparsers, "face", FaceAnalysisCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
