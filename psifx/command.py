import argparse
import argcomplete


class Command(object):
    """Interface for defining commands.

    Command instances must implement the `setup()` method, and they should
    implement the `execute()` method if they perform any functionality beyond
    defining subparsers.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """Setup the command-line arguments for the command.

        Args:
            parser: an `argparse.ArgumentParser` instance
        """
        raise NotImplementedError("subclass must implement setup()")

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """Executes the command on the given args.

        args:
            parser: the `argparse.ArgumentParser` instance for the command
            args: an `argparse.Namespace` instance containing the arguments
                for the command
        """
        raise NotImplementedError("subclass must implement execute()")


def has_subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return True

    return False


def iter_subparsers(parser: argparse.ArgumentParser):
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
        print("\n%s\n%s" % ("*" * 79, parser.format_help()))
        for subparser in iter_subparsers(parser):
            RecursiveHelpAction._recurse(subparser)


def register_command(
    parent: argparse.ArgumentParser, name: str, command: Command, recursive_help=True
) -> argparse.ArgumentParser:
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
    command: Command, version: str = None, recursive_help: bool = True
) -> argparse.ArgumentParser:
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

    argcomplete.autocomplete(parser)
    return parser


class PsifxCommand(Command):
    """The psifx command-line interface."""

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")

        from psifx.video.command import VideoCommand
        from psifx.audio.command import AudioCommand

        register_command(subparsers, "video", VideoCommand)
        register_command(subparsers, "audio", AudioCommand)

        # parser.add_argument(
        #     "--overwrite",
        #     default=False,
        #     action=argparse.BooleanOptionalAction,
        #     help="Overwrite existing files, otherwise raises an error.",
        # )
        # parser.add_argument(
        #     "--verbose",
        #     default=True,
        #     action=argparse.BooleanOptionalAction,
        #     help="Verbosity of the script.",
        # )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()


def main():
    parser = register_main_command(PsifxCommand, version="0.0.0")
    args = parser.parse_args()
    args.execute(args)
