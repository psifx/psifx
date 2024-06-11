import argparse, json
import os

from psifx.io.yaml import YAMLReader
from psifx.text.instruction.tool import InstructionTool
from psifx.utils.command import Command, register_command
from psifx.text.chat.tool import ChatTool
from psifx.text.llm.command import AddLLMArgument
class InstructionCommand(Command):
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
            '--input',
            type=str,
            required=True,
            help="path to the input .csv file")
        parser.add_argument(
            '--output',
            type=str,
            required=True,
            help="path to the output .csv file")
        parser.add_argument(
            '--instruction',
            type=str,
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_instruction.yaml'),
            help="instruction or path to a .yaml file containing the prompt template and parameters for segmentation")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        InstructionTool(
            model=args.model,
            overwrite=args.overwrite,
            verbose=args.verbose,
            **YAMLReader.read(args.instruction)
        ).segment_csv(
            input_path= args.input,
            output_path=args.output
        )

