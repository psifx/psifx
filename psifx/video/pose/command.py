import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.command import MediaPipeCommand
from psifx.video.pose.tool import PoseEstimationTool


class PoseEstimationCommand(Command):
    """
    Tools for estimating human poses from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "visualization", VisualizationCommand)
        register_command(subparsers, "mediapipe", MediaPipeCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class VisualizationCommand(Command):
    """
    Tool for visualizing the poses over the video.
    Expected video input extension: path/to/video.{any ffmpeg readable}
    Expected pose input extension: path/to/poses.tar
    Expected visualization video output extension: path/to/visualization-video.{any ffmpeg readable}
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
            help="path to the input video",
        )
        parser.add_argument(
            "--poses",
            type=Path,
            required=True,
            help="path to the pose archive",
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the visualization video",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0.0,
            help="threshold for not displaying low confidence keypoints",
        )
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
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
