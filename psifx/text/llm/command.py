import argparse, json

from psifx.utils.command import Command, register_command
from psifx.text.llm.ollama.command import OllamaCommand
from psifx.text.llm.hf.command import HFCommand

class LLMCommand(Command):
    """
    Tool for a llm
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")
        register_command(subparsers, "hf", HFCommand)
        register_command(subparsers, "ollama", OllamaCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()

def AddLLMArgument(parser):
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='name of the model, or path to a .json file containing the model specification')