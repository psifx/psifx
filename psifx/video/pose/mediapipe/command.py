import argparse
from pathlib import Path

from psifx.utils.command import Command, register_command
from psifx.video.pose.mediapipe.tool import (
    MediaPipePoseEstimationTool,
    MediaPipePoseEstimationAndSegmentationTool,
)


class MediaPipeCommand(Command):
    """
    Tool for running MediaPipe.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        from psifx.video.pose.command import VisualizationCommand

        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "inference", MediaPipeInferenceCommand)
        register_command(subparsers, "visualization", VisualizationCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


class MediaPipeInferenceCommand(Command):
    """
    Tool for inferring human pose with MediaPipe Holistic.
    Expected video input extension: path/to/video.{any ffmpeg readable}
    Expected pose output extension: path/to/poses.tar
    Expected mask input extension (optional): path/to/masks.{any ffmpeg readable}
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
            "--poses",
            type=Path,
            required=True,
            help="path to the pose archive",
        )
        parser.add_argument(
            "--masks",
            type=Path,
            default=None,
            help="path to the binary mask video",
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
