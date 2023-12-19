import argparse

from psifx.utils.command import Command, register_command
from psifx.video.face.openface.command import OpenFaceCommand


class FaceAnalysisCommand(Command):
    """
    Tools for estimating face features from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "openface", OpenFaceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
