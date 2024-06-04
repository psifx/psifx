import argparse, json

from psifx.utils.command import Command, register_command
from psifx.text.llm.hf.tool import get_transformers_pipeline


class HFCommand(Command):
    """
    Tool for getting a transformers pipeline
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
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
        parser.add_argument(
            "--overwrite",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="overwrite existing files, otherwise raises an error",
        )
        parser.add_argument(
            "--verbose",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="verbosity of the script",
        )
        parser.set_defaults(llm=lambda args: get_transformers_pipeline(
            args.model,
            quantization=args.quantization,
            device_map=args.device_map,
            model_kwargs=args.model_kwargs,
            max_new_tokens=args.max_new_tokens,
            pipeline_kwargs=args.pipeline_kwargs)
                            )

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
