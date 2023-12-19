import argparse

from psifx.utils.command import Command, register_command


class SpeechCommand(Command):
    """
    Tools for extracting non-verbal speech features from an audio track.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.audio.speech.opensmile.command import OpenSmileCommand

        register_command(subparsers, "opensmile", OpenSmileCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
