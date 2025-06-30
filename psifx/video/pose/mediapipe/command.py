"""MediaPipe pose estimation command-line interface."""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.tool import MediaPipePoseEstimationTool
from psifx.video.tracking.tool import TrackingTool


class MediaPipeCommand(Command):
    """
    Command-line interface for running MediaPipe.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        from psifx.video.pose.command import VisualizationCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "single-inference", MediaPipeSingleInferenceCommand)
        register_command(subparsers, "multi-inference", MediaPipeMultiInferenceCommand)
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


class MediaPipeSingleInferenceCommand(Command):
    """
    Command-line interface for inferring human pose with MediaPipe Holistic.
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
            help="path to the output pose archive, such as ``/path/to/poses.tar.gz``",
        )
        parser.add_argument(
            "--mask",
            type=Path,
            help="path to the input .mp4 mask file",
        )
        parser.add_argument(
            "--mask_threshold",
            type=float,
            default=0.1,
            help="threshold for the binarization of the segmentation mask",
        )
        parser.add_argument(
            "--model_complexity",
            type=int,
            default=2,
            help="complexity of the model: {0, 1, 2}, higher means more FLOPs, "
                 "but also more accurate results",
        )
        parser.add_argument(
            "--smooth",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="temporally smooth the inference results to reduce the jitter",
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

        def use_mediapipe(args, video_path):
            mediapipe_tool = MediaPipePoseEstimationTool(
                model_complexity=args.model_complexity,
                smooth=args.smooth,
                device=args.device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            mediapipe_tool.inference(
                video_path=video_path,
                poses_path=args.poses,
            )
            del mediapipe_tool

        if args.mask is None:
            use_mediapipe(args, args.video)
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
                use_mediapipe(args, tmp_dir / args.mask.name)


class MediaPipeMultiInferenceCommand(Command):
    """
    Command-line interface for inferring human pose with MediaPipe Holistic.
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
            "--poses_dir",
            type=Path,
            required=True,
            help="path to the output pose directory, such as ``/path/to/poses``",
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
            "--mask_threshold",
            type=float,
            default=0.1,
            help="threshold for the binarization of the segmentation mask",
        )
        parser.add_argument(
            "--model_complexity",
            type=int,
            default=2,
            help="complexity of the model: {0, 1, 2}, higher means more FLOPs, "
                 "but also more accurate results",
        )
        parser.add_argument(
            "--smooth",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="temporally smooth the inference results to reduce the jitter",
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

        if args.poses_dir.exists() and any(args.poses_dir.iterdir()):
            if args.overwrite:
                print(f"Poses directory {args.poses_dir} is non-empty")
            else:
                raise FileExistsError(f"Poses directory {args.poses_dir} is non-empty")

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

        mediapipe_tool = MediaPipePoseEstimationTool(
            model_complexity=args.model_complexity,
            smooth=args.smooth,
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
                mediapipe_tool.inference(
                    video_path=tmp_dir / mask.name,
                    poses_path=args.poses_dir / f"{mask.stem}.tar.gz",
                )
        del tracking_tool
        del mediapipe_tool
