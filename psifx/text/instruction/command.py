"""text instruction command-line interface."""

import argparse
from psifx.text.instruction.tool import InstructionTool
from psifx.utils.command import Command
from psifx.text.llm.command import instantiate_llm_tool, instantiate_llm, add_llm_argument
from pathlib import Path


class InstructionCommand(Command):
    """
    Command-line interface for custom instructions
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
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
            type=Path,
            required=True,
            help="path to the input .txt, .csv or .vtt file")
        parser.add_argument(
            '--output',
            type=Path,
            required=True,
            help="path to the output .txt or .csv file")

        add_llm_argument(parser)

        parser.add_argument(
            '--instruction',
            type=Path,
            required=True,
            help="Path to a .yaml file containing the prompt and parser.")

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        llm_tool = instantiate_llm_tool(args)
        llm = instantiate_llm(args, llm_tool)
        chain = llm_tool.chain_from_yaml(llm, args.instruction)

        tool = InstructionTool(
            overwrite=args.overwrite,
            verbose=args.verbose,
            chain=chain
        )

        path = Path(args.input)
        if path.suffix == '.txt':
            tool.apply_to_txt(input_path=args.input,
                              output_path=args.output)
        elif path.suffix == '.csv':
            tool.apply_to_csv(input_path=args.input,
                              output_path=args.output)
        elif path.suffix == '.vtt':
            tool.apply_to_vtt(input_path=args.input,
                              output_path=args.output)
        else:
            raise NameError(f"Input path should be .txt, .csv or .vtt. Got {args.input} instead.")
