import argparse, json

from psifx.utils.command import Command, register_command
from psifx.text.llm.ollama.command import OllamaCommand
from psifx.text.llm.hf.command import HFCommand
from psifx.text.llm.openai.command import OpenAICommand
from psifx.text.llm.anthropic.command import AnthropicCommand
class LLMCommand(Command):
    """
    Command-line interface for instantiating a llm.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(title="available commands")
        register_command(subparsers, "hf", HFCommand)
        register_command(subparsers, "ollama", OllamaCommand)
        register_command(subparsers, "openai", OpenAICommand)
        register_command(subparsers, "anthropic", AnthropicCommand)
    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()

def AddLLMArgument(parser):
    parser.add_argument(
        '--llm',
        type=str,
        required=True,
        help='path to a .yaml file containing the large language model specifications')