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
        '--llm',
        type=str,
        required=True,
        help='path to a .yaml file containing the large language model specifications')