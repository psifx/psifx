import argparse
from psifx.text.instruction.tool import InstructionTool
from psifx.text.llm.tool import LLMUtility
from psifx.utils.command import Command
from psifx.text.llm.command import AddLLMArgument


class InstructionCommand(Command):
    """
    Command-line interface for custom instructions
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
            required=True,
            help="path to a .yaml file containing the prompt and parser")

        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        llm = LLMUtility.llm_from_yaml(args.llm)
        chains = LLMUtility.chains_from_yaml(llm, args.instruction)
        InstructionTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
            chain=next(iter(chains.values()))
        ).apply_to_csv(
            input_path=args.input,
            output_path=args.output
        )
