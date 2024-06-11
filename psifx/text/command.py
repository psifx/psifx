"""text processing command-line interface."""

import argparse


from psifx.utils.command import Command, register_command
from psifx.text.chat.command import ChatCommand
from psifx.text.llm.command import LLMCommand
from psifx.text.instruction.command import InstructionCommand
from psifx.text.tasc.command import TascCommand


class TextCommand(Command):
    """
    Command-line interface for processing text.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "chat", ChatCommand)
        register_command(subparsers, "llm", LLMCommand)
        register_command(subparsers, "instruction", InstructionCommand)
        register_command(subparsers, "tasc", TascCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
