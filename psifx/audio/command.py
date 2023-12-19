import argparse

from psifx.utils.command import Command, register_command
from psifx.audio.diarization.command import DiarizationCommand
from psifx.audio.identification.command import IdentificationCommand
from psifx.audio.manipulation.command import ManipulationCommand
from psifx.audio.speech.command import SpeechCommand
from psifx.audio.transcription.command import TranscriptionCommand


class AudioCommand(Command):
    """
    Tools for processing audio tracks.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "diarization", DiarizationCommand)
        register_command(subparsers, "identification", IdentificationCommand)
        register_command(subparsers, "manipulation", ManipulationCommand)
        register_command(subparsers, "speech", SpeechCommand)
        register_command(subparsers, "transcription", TranscriptionCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
