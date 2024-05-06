"""command utilities."""

import argparse
from typing import Type


class Command:
    """
    Base class for defining commands.

    Command instances must implement the ``setup()`` method, and they should
    implement the ``execute()`` method if they perform any functionality beyond
    defining subparsers.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        """Setup the command-line arguments for the command.

        Args:
            parser: an ``argparse.ArgumentParser`` instance
        """
        raise NotImplementedError("subclass must implement setup()")

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        """Executes the command on the given args.

        args:
            parser: the ``argparse.ArgumentParser`` instance for the command
            args: an ``argparse.Namespace`` instance containing the arguments
                for the command
        """
        raise NotImplementedError("subclass must implement execute()")


def has_subparsers(parser: argparse.ArgumentParser) -> bool:
    """
    Checks whether the parser had subparsers.

    :param parser:
    :return:
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return True

    return False


def iter_subparsers(parser: argparse.ArgumentParser):
    """
    Iterates through the subparsers.

    :param parser:
    :return:
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield subparser


class RecursiveHelpAction(argparse._HelpAction):
    def __call__(self, parser: argparse.ArgumentParser, *args, **kwargs):
        self._recurse(parser)
        parser.exit()

    @staticmethod
    def _recurse(parser: argparse.ArgumentParser):
        print("", "*" * 79, parser.format_help(), sep="\n")
        for subparser in iter_subparsers(parser):
            RecursiveHelpAction._recurse(subparser)


def register_command(
    parent: argparse.ArgumentParser,
    name: str,
    command: Type[Command],
    recursive_help=True,
) -> argparse.ArgumentParser:
    """
    Registers a command to a parent subparser and returns the newly created parser.

    :param parent:
    :param name:
    :param command:
    :param recursive_help:
    :return:
    """
    parser = parent.add_parser(
        name,
        help=command.__doc__.splitlines()[0],
        description=command.__doc__.rstrip(),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.set_defaults(execute=lambda args: command.execute(parser, args))
    command.setup(parser)

    if recursive_help and has_subparsers(parser):
        parser.add_argument(
            "--all-help",
            action=RecursiveHelpAction,
            help="show help recursively and exit",
        )

    return parser


def register_main_command(
    command: Type[Command],
    version: str = None,
    recursive_help: bool = True,
) -> argparse.ArgumentParser:
    """
    Registers the main command entrypoint and returns the parser.

    :param command:
    :param version:
    :param recursive_help:
    :return:
    """
    parser = argparse.ArgumentParser(description=command.__doc__.rstrip())

    parser.set_defaults(execute=lambda args: command.execute(parser, args))
    command.setup(parser)

    if version:
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=version,
            help="show version info",
        )

    if recursive_help and has_subparsers(parser):
        parser.add_argument(
            "--all-help",
            action=RecursiveHelpAction,
            help="show help recursively and exit",
        )

    return parser
