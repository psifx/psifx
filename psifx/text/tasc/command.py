import argparse, json
import os

from psifx.utils.command import Command, register_command
from psifx.text.tasc.tool import TascTool
from psifx.text.llm.command import AddLLMArgument


class TascCommand(Command):
    """
    Tool for TASc
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
            required=True,
            help="name of the speaker for .vtt transcription or name of column for .csv file to apply segmentation on")
        parser.add_argument(
            '--instruction',
            type=str,
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_instruction.txt'),
            help="instruction or path to a .txt file containing the instruction")
        parser.add_argument(
            '--separator',
            type=str,
            default='//',
            help="separator to use for parsing the llm generation")
        parser.add_argument(
            '--flag',
            type=str,
            default='Segmentation:',
            help="start flag to use for parsing the llm generation")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        TascTool(
            model=args.model,
            instruction=args.instruction,
            start_flag=args.flag,
            separator=args.separator,
            overwrite=args.overwrite,
            verbose=args.verbose
        ).segment(
            transcription_path=args.transcription,
            segmented_transcription_path=args.segmentation,
            speaker=args.speaker
        )
