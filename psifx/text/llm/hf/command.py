import argparse, json

from psifx.utils.command import Command, register_command
from psifx.text.llm.hf.tool import get_lc_hf


class HFCommand(Command):
    """
    Command-line interface for HuggingFace Transformers LLM.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        """
        Sets up the command.

        :param parser: The argument parser.
        :return:
        """
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Name of the model')

        parser.add_argument(
            '--quantization',
            type=str,
            default=None,
            help='Quantization method (default: None)')

        parser.add_argument(
            '--token',
            type=str,
            default=None,
            help='HuggingFace token (default: None)')

        parser.add_argument(
            '--max_new_tokens',
            type=int,
            default=10,
            help='Maximum number of new tokens (default: 10)')

        parser.add_argument(
            '--model_kwargs',
            type=json.loads,
            default=None,
            help='Additional model keyword arguments as JSON string')

        parser.add_argument(
            '--pipeline_kwargs',
            type=json.loads,
            default=None,
            help='Additional pipeline keyword arguments as JSON string')

        parser.add_argument(
            "--api_token",
            type=str,
            default=None,
            help="API token for downloading the models from HuggingFace",
        )
        parser.set_defaults(llm=lambda args: get_lc_hf(**args))

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
