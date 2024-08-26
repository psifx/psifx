"""large language model command-line interface."""

import argparse
import os
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
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        subparsers = parser.add_subparsers(title="available commands")
        register_command(subparsers, "hf", HFCommand)
        register_command(subparsers, "ollama", OllamaCommand)
        register_command(subparsers, "openai", OpenAICommand)
        register_command(subparsers, "anthropic", AnthropicCommand)

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        """
        Executes the command.

        :param parser: The argument parser.
        :param args: The arguments.
        :return:
        """
        parser.print_help()


def add_llm_argument(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for large language model.
    :param parser: An argument parser.
    """
    parser.add_argument(
        '--llm',
        type=yaml,
        default='small-local',
        required=False,
        help="the large language model to use, can be 'small-local','medium-local', 'large-local', 'openai', 'anthropic' or path to a .yaml configuration file (default small-local)")


def yaml(model: str, directory: str = 'configs') -> str:
    """
    Convert the model to a pre-made yaml file path (if it is not already a yaml path).
    :param model: The name of a pre-made config or a yaml path.
    :param directory: The directory that contains the pre-made configs.
    :return: The path to a yaml config file.
    """
    if model == 'small-local':
        print(
            'You are using the small-local model. Better results can be obtained with larger alternatives. See documentation for details.')
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    if os.path.isfile(model):
        return model
    yaml_file_path = os.path.join(directory, f"{model}.yaml")
    if os.path.isfile(yaml_file_path):
        return yaml_file_path

    available_files = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.yaml')]

    raise FileNotFoundError(
        f"No corresponding .yaml file found for '{model}' in '{directory}'.\n"
        f"Available options: {', '.join(available_files)}"
    )
