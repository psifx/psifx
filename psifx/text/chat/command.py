import argparse, json

from psifx.text.llm.tool import LLMUtility
from psifx.utils.command import Command, register_command
from psifx.text.chat.tool import ChatTool
from psifx.text.llm.command import AddLLMArgument
class ChatCommand(Command):
    """
    Tool for getting a transformers pipeline
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
            required=True,
            help='prompt or path to a .txt file containing the prompt')
        AddLLMArgument(parser)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        llm = LLMUtility.llm_from_yaml(args.llm)
        ChatTool(llm).chat(args.prompt)
