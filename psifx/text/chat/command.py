"""chatbot command-line interface."""

import argparse
from psifx.utils.command import Command
from psifx.text.chat.tool import ChatTool
from psifx.text.llm.command import add_llm_argument, instantiate_llm_tool, instantiate_llm
from pathlib import Path

class ChatCommand(Command):
    """
    Command-line interface for a chatbot
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
            '--prompt',
            type=str,
            default="",
            help='prompt or path to a .txt file containing the prompt')
        parser.add_argument(
            '--output',
            type=Path,
            default=None,
            help='path to a .txt save file')
        add_llm_argument(parser)

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
        ChatTool(
            llm=llm,
            overwrite=args.overwrite,
            verbose=args.verbose
        ).chat(
            prompt=args.prompt,
            save_path=args.output)
