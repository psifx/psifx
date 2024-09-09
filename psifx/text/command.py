"""text processing command-line interface."""

import argparse


from psifx.utils.command import Command, register_command
from psifx.text.chat.command import ChatCommand
from psifx.text.instruction.command import InstructionCommand


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
        register_command(subparsers, "instruction", InstructionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
