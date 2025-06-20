"""tracking command-line interface."""

import argparse
import os
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.tracking.samurai.command import SamuraiCommand
from psifx.video.tracking.tool import TrackingTool


class TrackingCommand(Command):
    """
    Command-line interface for processing videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "samurai", SamuraiCommand)
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


class VisualizationTrackingCommand(Command):
    """
    Command-line interface for the visualization of tracking.
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

        parser.add_argument(
            "--visualization",
            type=Path,
            required=True,
            help="path to the output visualization video file, such as ``/path/to/visualization.mp4`` (or .avi, .mkv, etc.)",
        )

        parser.add_argument(
            "--blackout",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="whether to black out the background (non-mask regions)",
        )

        parser.add_argument(
            "--labels",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="whether to add labels",
        )

        parser.add_argument(
            "--color",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="whether to color the masks",
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
        tool = TrackingTool(
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        mask_files = []
        for masks in args.masks:
            if os.path.isdir(masks):
                mask_files += [f for f in masks.iterdir()]
            else:
                mask_files.append(masks)

        tool.visualize(
            video_path=args.video,
            mask_paths=mask_files,
            visualization_path=args.visualization,
            blackout=args.blackout,
            color=args.color,
            labels=args.labels,
        )
        del tool
