import argparse

from psifx.command import Command, register_command


class VideoCommand(Command):
    """
    Tools for processing videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.video.pose.command import PoseEstimationCommand
        from psifx.video.manipulation.command import ManipulationCommand
        from psifx.video.face.command import FaceAnalysisCommand

        register_command(subparsers, "pose", PoseEstimationCommand)
        register_command(subparsers, "manipulation", ManipulationCommand)
        register_command(subparsers, "face", FaceAnalysisCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
