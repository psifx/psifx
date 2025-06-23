"""samurai tracking command-line interface."""

import argparse
from pathlib import Path

import torch

from psifx.utils.command import Command, register_command
from psifx.video.tracking.samurai.tool import SamuraiTrackingTool


class SamuraiCommand(Command):
    """
    Command-line interface for running Samurai.
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

        register_command(subparsers, "inference", SamuraiInferenceCommand)
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





class SamuraiInferenceCommand(Command):
    """
    Command-line interface for tracking video elements with Samurai.
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
            "--model_size",
            type=str,
            choices=["tiny", "small", "base_plus", "large"],
            default="tiny",
            help="size of the sam-2 model",
        )
        parser.add_argument(
            "--yolo_model",
            type=str,
            choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
            default="yolo11n.pt",
            help="name of the yolo model",
        )
        parser.add_argument(
            "--object_class",
            type=int,
            default=0,
            help="class of the object to detect according to yolo (0 for people)",
        )
        parser.add_argument(
            "--max_objects",
            type=int,
            default=None,
            help="maximum number of people/objects to detect",
        )
        parser.add_argument(
            "--step",
            type=int,
            default=30,
            help="step size in frames to perform object detection",
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
        tool = SamuraiTrackingTool(
            model_size=args.model_size,
            use_samurai=True,
            yolo_model=args.yolo_model,
            device=args.device,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.infer(
            video_path=args.video,
            mask_dir=args.mask_dir,
            object_class=args.object_class,
            max_objects=args.max_objects,
            step_size=args.step
        )
        del tool
