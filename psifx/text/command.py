import argparse

from psifx.utils.command import Command, register_command
from psifx.text.chat.command import ChatCommand
from psifx.text.llm.command import LLMCommand
from psifx.text.tasc.command import TascCommand

class TextCommand(Command):
    """
    Tools for processing text.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "chat", ChatCommand)
        register_command(subparsers, "llm", LLMCommand)
        register_command(subparsers, "tasc", TascCommand)
    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
