"""OpenFace face analysis command-line interface."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

from psifx.utils.command import Command, register_command
from psifx.video.face.openface.tool import OpenFaceTool
from psifx.video.tracking.tool import TrackingTool


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

        register_command(subparsers, "single-inference", OpenFaceSingleInferenceCommand)
        register_command(subparsers, "multi-inference", OpenFaceMultiInferenceCommand)
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


class OpenFaceSingleInferenceCommand(Command):
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
            "--mask",
            type=Path,
            help="path to the input .mp4 mask file",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
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
        if args.mask is None:
            openface_tool = OpenFaceTool(
                device=args.device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            openface_tool.inference(
                video_path=args.video,
                features_path=args.features,
            )
            del openface_tool
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)

                tracking_tool = TrackingTool(
                    device=args.device,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )

                tracking_tool.visualize(
                    video_path=args.video,
                    mask_paths=args.mask,
                    visualization_path=tmp_dir / args.mask.name,
                    blackout=True,
                    color=False,
                    labels=False,
                )
                del tracking_tool
                openface_tool = OpenFaceTool(
                    device=args.device,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )
                openface_tool.inference(
                    video_path=tmp_dir / args.mask.name,
                    features_path=args.features,
                )
                del openface_tool


class OpenFaceMultiInferenceCommand(Command):
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
            "--masks",
            type=Path,
            nargs='+',
            required=True,
            help="list of path to mask directories or individual .mp4 mask files",
        )

        def custom_error(message):
            parser.print_usage(sys.stderr)
            args = {'prog': parser.prog, 'message': message}
            sys.stderr.write('%(prog)s: error: %(message)s\n' % args)
            if 'masks' in message:
                sys.stderr.write("run psifx tracking samurai inference to get masks\n")
            parser.exit(2)

        parser.error = custom_error

        parser.add_argument(
            "--features_dir",
            type=Path,
            required=True,
            help="path to the output feature directory, such as ``/path/to/openface``",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
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

        if args.features_dir.exists() and any(args.features_dir.iterdir()):
            if args.overwrite:
                print(f"Features directory {args.features_dir} is non-empty")
            else:
                raise FileExistsError(f"Features directory {args.features_dir} is non-empty")

        mask_files = []
        for mask in args.masks:
            if mask.is_dir():
                dir_files = list(mask.iterdir())
                non_mp4_files = [str(f) for f in dir_files if not (f.is_file() and f.suffix == '.mp4')]
                if non_mp4_files:
                    raise ValueError(f"Directory {mask} contains non-.mp4 files: {non_mp4_files}")
                mask_files.extend(dir_files)
            elif mask.is_file() and mask.suffix == '.mp4':
                mask_files.append(mask)
            else:
                raise FileNotFoundError(f"{mask} is not a directory or a .mp4 file")

        tracking_tool = TrackingTool(
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        openface_tool = OpenFaceTool(
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for mask in mask_files:
                tracking_tool.visualize(
                    video_path=args.video,
                    mask_paths=[mask],
                    visualization_path=tmp_dir / mask.name,
                    blackout=True,
                    color=False,
                    labels=False,
                )
                openface_tool.inference(
                    video_path=tmp_dir / mask.name,
                    features_path=args.features_dir / f"{mask.stem}.tar.gz",
                )
        del tracking_tool
        del openface_tool


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
            nargs='+',
            help="list of path to the input feature directories or individual archive ``.tar.gz`` files",
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
        feature_files = []
        for features in args.features:
            if os.path.isdir(features):
                feature_files += [f for f in features.iterdir()]
            else:
                feature_files.append(features)

        tool = OpenFaceTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.visualization(
            video_path=args.video,
            features_path=feature_files,
            visualization_path=args.visualization,
            depth=args.depth,
            f_x=args.f_x,
            f_y=args.f_y,
            c_x=args.f_x,
            c_y=args.f_y,
        )
        del tool
