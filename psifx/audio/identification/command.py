import argparse

from psifx.command import Command, register_command


class IdentificationCommand(Command):
    """
    Tools for identifying speakers in audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.audio.identification.pyannote.command import PyannoteCommand

        register_command(subparsers, "pyannote", PyannoteCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
