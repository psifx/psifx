import argparse
import os

from psifx.utils.command import Command
from psifx.text.tasc.segment.tool import SegmentTool
from psifx.text.llm.command import AddLLMArgument
from psifx.io.yaml import YAMLReader

class SegmentCommand(Command):
    """
    Command-line interface for TASc segmentation.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
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
        parser.add_argument(
            '--transcription',
            type=str,
            required=True,
            help="path to the transcription .vtt or .csv file")
        parser.add_argument(
            '--segmentation',
            type=str,
            required=True,
            help="path to the output segmented transcription .vtt or .csv file")
        parser.add_argument(
            '--speaker',
            type=str,
            default=None,
            help="name of the speaker for .vtt or .csv transcription")
        parser.add_argument(
            '--instruction',
            type=str,
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_instruction.yaml'),
            help="instruction or path to a .yaml file containing the prompt template and parameters for segmentation")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        SegmentTool(
            model=args.model,
            overwrite=args.overwrite,
            verbose=args.verbose,
            **YAMLReader.read(args.instruction)
        ).use(
            transcription_path=args.transcription,
            segmented_transcription_path=args.segmentation,
            speaker=args.speaker
        )
