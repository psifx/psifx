"""pose estimation command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.command import MediaPipeCommand
from psifx.video.pose.tool import PoseEstimationTool


class PoseEstimationCommand(Command):
    """
    Command-line interface for estimating human poses from videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "mediapipe", MediaPipeCommand)
        register_command(subparsers, "visualization", VisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class VisualizationCommand(Command):
    """
    Command-line interface for visualizing the poses over the video.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            "--video",
            type=Path,
            required=True,
            help="path to the input video file, such as ``/path/to/video.mp4`` (or .avi, .mkv, etc.)",
        )
        parser.add_argument(
            "--poses",
            type=Path,
            required=True,
            help="path to the input pose archive, such as ``/path/to/poses.tar.gz``",
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the output visualization video file, such as ``/path/to/visualization.mp4`` (or .avi, .mkv, etc.)",
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
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
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
