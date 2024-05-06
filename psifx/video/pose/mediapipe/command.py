"""MediaPipe pose estimation command-line interface."""

import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.tool import (
    MediaPipePoseEstimationTool,
    MediaPipePoseEstimationAndSegmentationTool,
)


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

        register_command(subparsers, "inference", MediaPipeInferenceCommand)
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


class MediaPipeInferenceCommand(Command):
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
            "--masks",
            type=Path,
            default=None,
            help="path to the output segmentation mask video file, such as ``/path/to/masks.mp4`` (or .avi, .mkv, etc.)",
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
            default="cpu",
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
        if args.masks is None:
            tool = MediaPipePoseEstimationTool(
                model_complexity=args.model_complexity,
                smooth=args.smooth,
                device=args.device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            tool.inference(
                video_path=args.video,
                poses_path=args.poses,
            )
        else:
            tool = MediaPipePoseEstimationAndSegmentationTool(
                model_complexity=args.model_complexity,
                smooth=args.smooth,
                mask_threshold=args.mask_threshold,
                device=args.device,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            tool.inference(
                video_path=args.video,
                poses_path=args.poses,
                masks_path=args.masks,
            )
        del tool
