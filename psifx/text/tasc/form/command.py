import argparse
import os

from psifx.io.yaml import YAMLReader
from psifx.text.tasc.form.tool import FormTool
from psifx.utils.command import Command
from psifx.text.llm.command import AddLLMArgument


class FormCommand(Command):
    """
    Command-line interface for TASc form.
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
            '--segment',
            type=str,
            required=True,
            help="path to the input segmented transcription .vtt or .csv file")
        parser.add_argument(
            '--form',
            type=str,
            required=True,
            help="path to the output form transcription .vtt or .csv file")
        parser.add_argument(
            '--speaker',
            type=str,
            default=None,
            help="name of the speaker for .vtt or .csv transcription")
        parser.add_argument(
            '--instruction',
            type=str,
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_instruction.yaml'),
            help="instruction or path to a .yaml file containing the instruction")
        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        FormTool(
            model=args.model,
            overwrite=args.overwrite,
            verbose=args.verbose,
            **YAMLReader.read(args.instruction)
        ).use(
            transcription_path=args.transcription,
            segmented_transcription_path=args.segmentation,
            speaker=args.speaker
        )

