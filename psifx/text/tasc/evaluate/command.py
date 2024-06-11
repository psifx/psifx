import argparse, json
import os

from psifx.utils.command import Command, register_command
from psifx.text.llm.command import AddLLMArgument


class EvaluateCommand(Command):
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
            '--marker',
            type=str,
            required=True,
            help="path to the input marker .vtt or .csv file")
        parser.add_argument(
            '--results',
            type=str,
            required=True,
            help="path to the output results .txt file")
        parser.add_argument(
            '--speaker',
            type=str,
            default=None,
            help="name of the speaker for .vtt or .csv transcription")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        pass