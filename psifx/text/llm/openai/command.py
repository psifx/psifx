import argparse
from typing import Union, Tuple, Any, Optional

from psifx.utils.command import Command
from psifx.text.llm.openai.tool import get_openai


class OpenAICommand(Command):
    """
    Command-line interface for OpenAI LLM.
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
            default="gpt-4o",
            help='Model name to use (default: "gpt-4o")')

        parser.add_argument(
            '--temperature',
            type=float,
            default=0,
            help='The temperature of the model. Increasing the temperature will make the model answer more creatively. (default: 0)'
        )

        parser.add_argument(
            '--timeout',
            type=Union[float, Tuple[float, float], Any, None],
            default=None,
            help='Timeout for requests. (default: None)'
        )

        parser.add_argument(
            '--max_tokens',
            type=Optional[int],
            default=None,
            help='Max number of tokens to generate. (default:None)'
        )
        parser.add_argument(
            '--max_retries',
            type=int,
            default=2,
            help='Max number of retries. (default:2)'
        )
        parser.add_argument(
            '--api_key',
            type=Optional[str],
            default=None,
            help='OpenAI API key. (default:None)'
        )

        parser.set_defaults(llm=lambda args: get_openai(**args))

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
