import argparse
from typing import Union

from psifx.utils.command import Command
from psifx.text.llm.anthropic.tool import get_anthropic


class AnthropicCommand(Command):
    """
    Command-line interface for Anthropic LLM.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--model',
            type=str,
            default="claude-3-5-sonnet-20240620",
            help='Model name to use (default: "claude-3-5-sonnet-20240620")')

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

        parser.set_defaults(llm=lambda args: get_anthropic(**args))

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
