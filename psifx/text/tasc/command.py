"""TASc command-line interface."""

import argparse

from psifx.utils.command import Command, register_command
from psifx.text.tasc.segment.command import SegmentCommand
from psifx.text.tasc.form.command import FormCommand
from psifx.text.tasc.marker.command import MarkerCommand
from psifx.text.tasc.evaluate.command import EvaluateCommand
from psifx.text.tasc.analysis.command import AnalysisCommand


class TascCommand(Command):
    """
    Command-line interface for TASc.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")

        register_command(subparsers, "segment", SegmentCommand)
        register_command(subparsers, "form", FormCommand)
        register_command(subparsers, "marker", MarkerCommand)
        register_command(subparsers, "evaluate", EvaluateCommand)
        register_command(subparsers, "analysis", AnalysisCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()
