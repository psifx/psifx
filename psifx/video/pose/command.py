import argparse
from pathlib import Path

from psifx.command import Command, register_command
from psifx.video.pose.tool import PoseEstimationTool


class PoseEstimationCommand(Command):
    """
    Tools for estimating human poses from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.video.pose.mediapipe.command import MediaPipeCommand

        register_command(subparsers, "visualization", VisualizationCommand)
        register_command(subparsers, "mediapipe", MediaPipeCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class VisualizationCommand(Command):
    """
    Tool for visualizing the poses over the video.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
        )
        parser.add_argument(
            "--poses",
            type=Path,
            required=True,
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Overwrite existing files, otherwise raises an error.",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Verbosity of the script.",
        )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        tool = PoseEstimationTool(
            device="cpu",
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.visualization(
            video_path=args.video,
            poses_path=args.poses,
            visualization_path=args.visualization,
            confidence_threshold=args.confidence_threshold,
        )
        del tool
