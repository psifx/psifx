"""``psifx`` command-line interface."""

import argparse

import psifx
from psifx.utils.command import Command, register_command, register_main_command
from psifx.video.command import VideoCommand
from psifx.audio.command import AudioCommand


class PsifxCommand(Command):
    """``psifx`` command-line interface."""

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "audio", AudioCommand)
        register_command(subparsers, "video", VideoCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


def get_parser() -> argparse.ArgumentParser:
    """
    Create a parser for the command-line interface.
    :return:
    """
    return register_main_command(PsifxCommand, version=psifx.__version__)


def main():
    """
    Entrypoint of the psifx command-line interface.
    :return:
    """
    parser = get_parser()
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
