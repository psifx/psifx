import argparse

import psifx
from psifx.utils.command import Command, register_command, register_main_command
from psifx.video.command import VideoCommand
from psifx.audio.command import AudioCommand


class PsifxCommand(Command):
    """psifx command-line interface."""

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "video", VideoCommand)
        register_command(subparsers, "audio", AudioCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


def main():
    """
    Entrypoint of the psifx package.
    :return:
    """
    parser = register_main_command(PsifxCommand, version=psifx.__version__)
    args = parser.parse_args()
    args.execute(args)
