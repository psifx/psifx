"""sam3 tracking command-line interface."""

import argparse
from pathlib import Path

import torch

from psifx.utils.command import Command, register_command
from psifx.video.tracking.sam3.tool import Sam3TrackingTool


class Sam3Command(Command):
    """
    Command-line interface for running SAM3.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        from psifx.video.tracking.command import VisualizationTrackingCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", Sam3InferenceCommand)
        register_command(subparsers, "visualization", VisualizationTrackingCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


class Sam3InferenceCommand(Command):
    """
    Command-line interface for tracking video elements with SAM3.
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
            "--mask_dir",
            type=Path,
            required=True,
            help="path to the output mask directory, such as ``/path/to/mask_dir``",
        )
        parser.add_argument(
            "--text_prompt",
            type=str,
            default="people",
            help="text description of objects to track (e.g., 'people', 'cars', 'dogs')",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=300,
            help="number of frames to process at once (lower values use less memory)",
        )
        parser.add_argument(
            "--iou_threshold",
            type=float,
            default=0.3,
            help="IoU threshold for stitching chunks together (0.0 to 1.0)",
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
        tool = Sam3TrackingTool(
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.infer(
            video_path=args.video,
            mask_dir=args.mask_dir,
            text_prompt=args.text_prompt,
            chunk_size=args.chunk_size,
            iou_threshold=args.iou_threshold,
        )
        del tool
