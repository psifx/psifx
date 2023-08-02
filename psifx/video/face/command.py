import argparse

from psifx.command import Command, register_command


class FaceAnalysisCommand(Command):
    """
    Tools for estimating face features from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.video.face.openface.command import OpenFaceCommand

        register_command(subparsers, "openface", OpenFaceCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
