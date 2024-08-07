import argparse

from psifx.text.llm.tool import LLMTool
from psifx.utils.command import Command
from psifx.text.chat.tool import ChatTool
from psifx.text.llm.command import AddLLMArgument


class ChatCommand(Command):
    """
    Command-line interface for a chatbot
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
            '--prompt',
            type=str,
            default="",
            help='prompt or path to a .txt file containing the prompt')
        parser.add_argument(
            '--output',
            type=str,
            default="",
            help='path to a .txt save file')
        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        llm = LLMTool().llm_from_yaml(args.llm)
        ChatTool(
            llm=llm,
            overwrite=args.overwrite,
            verbose=args.verbose
        ).chat(
            prompt=args.prompt,
            save_file=args.output)
