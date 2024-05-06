"""OpenFace face analysis command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.face.openface.tool import OpenFaceTool


class OpenFaceCommand(Command):
    """
    Command-line interface for running OpenFace.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", OpenFaceInferenceCommand)
        register_command(subparsers, "visualization", OpenFaceVisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class OpenFaceInferenceCommand(Command):
    """
    Command-line interface for inferring face features from videos with OpenFace.
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
            "--features",
            type=Path,
            required=True,
            help="path to the output feature archive, such as ``/path/to/openface.tar.gz``",
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
        tool = OpenFaceTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.inference(
            video_path=args.video,
            features_path=args.features,
        )
        del tool


class OpenFaceVisualizationCommand(Command):
    """
    Command-line interface for visualizing face features from videos with OpenFace.
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
            "--features",
            type=Path,
            required=True,
            help="path to the input feature archive, such as ``/path/to/openface.tar.gz``",
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the output video file, such as ``/path/to/visualization.mp4`` (or .avi, .mkv, etc.)",
        )
        parser.add_argument(
            "--depth",
            type=float,
            default=3.0,
            help="projection: assumed static depth of the subject in meters",
        )
        parser.add_argument(
            "--f_x",
            type=float,
            default=None,
            help="projection: x-axis of the focal length",
        )
        parser.add_argument(
            "--f_y",
            type=float,
            default=None,
            help="projection: y-axis of the focal length",
        )
        parser.add_argument(
            "--c_x",
            type=float,
            default=None,
            help="projection: x-axis of the principal point",
        )
        parser.add_argument(
            "--c_y",
            type=float,
            default=None,
            help="projection: y-axis of the principal point",
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
        tool = OpenFaceTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.visualization(
            video_path=args.video,
            features_path=args.features,
            visualization_path=args.visualization,
            depth=args.depth,
            f_x=args.f_x,
            f_y=args.f_y,
            c_x=args.f_x,
            c_y=args.f_y,
        )
        del tool
