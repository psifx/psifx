import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.face.openface.tool import OpenFaceTool


class OpenFaceCommand(Command):
    """
    Tool for running OpenFace.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", OpenFaceInferenceCommand)
        register_command(subparsers, "visualization", OpenFaceVisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class OpenFaceInferenceCommand(Command):
    """
    Tool for inferring face features from videos with OpenFace.
    Expected video input extension: path/to/video.{any ffmpeg readable}
    Expected OpenFace features output extension: path/to/openface-features.tar
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
            "--features",
            type=Path,
            required=True,
            help="path to the feature archive",
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
    Tool for visualizing face features from videos with OpenFace.
    Expected video input extension: path/to/video.{any ffmpeg readable}
    Expected OpenFace features input extension: path/to/openface-features.tar
    Expected visualization video output extension: path/to/visualization-features.{any ffmpeg readable}
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
            "--features",
            type=Path,
            required=True,
            help="path to the feature archive",
        )
        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the visualization video",
        )
        parser.add_argument(
            "--depth",
            type=float,
            default=3.0,
            help="projection: assumed static depth of the subject",
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
