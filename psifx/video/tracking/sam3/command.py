"""sam3 tracking command-line interface."""

import argparse
import os
from pathlib import Path

import torch

from psifx.utils.constants import SAM3_PATH
from psifx.utils.command import Command, register_command
from psifx.video.tracking.sam3.tool import (
    DEFAULT_MERGE_THRESHOLD,
    DEFAULT_MIN_FRAGMENT_FRAMES,
    DEFAULT_OVERLAP_CENTROID_THRESHOLD,
    DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD,
    Sam3TrackingTool,
)


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
            "--max_num_objects",
            type=int,
            default=None,
            help=(
                "optional cap on tracked object count "
                "(e.g., set to 2 for a two-person interaction)"
            ),
        )
        parser.add_argument(
            "--id_correction",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "merge the ids of a same person/object that tracking fragmented "
                "(occlusion, leaving and re-entering the frame, chunk boundary), "
                "using appearance; writes a second, id-corrected mask directory "
                "and leaves ``--mask_dir`` untouched"
            ),
        )
        parser.add_argument(
            "--merged_mask_dir",
            type=Path,
            default=None,
            help="path to the id-corrected mask directory, defaults to ``<mask_dir>_merged``",
        )
        parser.add_argument(
            "--merge_report",
            type=Path,
            default=None,
            help=(
                "path to the JSON report of every merge decision, "
                "defaults to ``<mask_dir>_merge_report.json``"
            ),
        )
        parser.add_argument(
            "--merge_threshold",
            type=float,
            default=DEFAULT_MERGE_THRESHOLD,
            help="minimum appearance similarity (0.0 to 1.0) to merge a track that ends with one that starts later",
        )
        parser.add_argument(
            "--min_fragment_frames",
            type=int,
            default=DEFAULT_MIN_FRAGMENT_FRAMES,
            help="ids with fewer non-empty frames than this are too short to be matched and are left as they are",
        )
        parser.add_argument(
            "--overlap_pixel_gap_threshold",
            type=float,
            default=DEFAULT_OVERLAP_PIXEL_GAP_THRESHOLD,
            help="maximum gap in pixels between two simultaneous masks to consider them parts of a same body",
        )
        parser.add_argument(
            "--overlap_centroid_threshold",
            type=float,
            default=DEFAULT_OVERLAP_CENTROID_THRESHOLD,
            help="maximum centroid distance in pixels between two simultaneous masks to consider them parts of a same body",
        )
        parser.add_argument(
            "--osnet",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "use OSNet appearance embeddings for id correction (needs the optional "
                "``torchreid`` package, falls back to color only when it is missing)"
            ),
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="device on which to run the inference, either 'cpu' or 'cuda'",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=SAM3_PATH,
            help="SAM3 model id or local path (e.g. 'facebook/sam3' or '/path/to/sam3')",
        )
        parser.add_argument(
            "--api_token",
            type=str,
            default=os.environ.get("HF_TOKEN"),
            help="Hugging Face token (defaults to HF_TOKEN env var when available)",
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
            model_path=args.model_path,
            api_token=args.api_token,
            max_num_objects=args.max_num_objects,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        tool.infer(
            video_path=args.video,
            mask_dir=args.mask_dir,
            text_prompt=args.text_prompt,
            chunk_size=args.chunk_size,
            iou_threshold=args.iou_threshold,
            id_correction=args.id_correction,
            merged_mask_dir=args.merged_mask_dir,
            merge_report_path=args.merge_report,
            id_correction_kwargs=dict(
                merge_threshold=args.merge_threshold,
                min_fragment_frames=args.min_fragment_frames,
                overlap_pixel_gap_threshold=args.overlap_pixel_gap_threshold,
                overlap_centroid_threshold=args.overlap_centroid_threshold,
                use_osnet=args.osnet,
            ),
        )
        del tool
