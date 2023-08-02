import argparse
from pathlib import Path

from psifx.command import Command, register_command
from psifx.video.manipulation.tool import ManipulationTool


class ManipulationCommand(Command):
    """
    Tool for manipulating videos.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "process", ProcessCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class ProcessCommand(Command):
    """
    Tool for processing videos.
    The trimming, cropping and resizing can be performed all at once, and in that order.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--in_video",
            type=Path,
            required=True,
        )
        parser.add_argument(
            "--out_video",
            type=Path,
            required=True,
        )
        parser.add_argument(
            "--start",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--end",
            type=float,
            default=None,
        )
        parser.add_argument(
            "--x_min",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--y_min",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--x_max",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--y_max",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--width",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--height",
            type=int,
            default=None,
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
        tool = ManipulationTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.process(
            in_video_path=args.in_video,
            out_video_path=args.out_video,
            start=args.start,
            end=args.end,
            x_min=args.x_min,
            y_min=args.y_min,
            x_max=args.x_max,
            y_max=args.y_max,
            width=args.width,
            height=args.height,
        )
        del tool
