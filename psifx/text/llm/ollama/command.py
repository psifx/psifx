import argparse
from typing import Union

from psifx.utils.command import Command
from psifx.text.llm.ollama.tool import get_ollama


class OllamaCommand(Command):
    """
    Command-line interface for Ollama LLM.
    """

    @staticmethod
    def setup(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--model',
            type=str,
            default="llama3",
            help='Model name to use (default: "llama3.1")')

        parser.add_argument(
            '--base_url',
            type=str,
            default="http://localhost:11434",
            help='Base url the model is hosted under (default: "http://localhost:11434")'
        )

        parser.add_argument(
            '--mirostat',
            type=int,
            default=0,
            help='Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)'
        )

        parser.add_argument(
            '--mirostat_eta',
            type=float,
            default=0.1,
            help='Influences how quickly the algorithm responds to feedback from the generated text. (default: 0.1)'
        )

        parser.add_argument(
            '--mirostat_tau',
            type=float,
            default=5.0,
            help='Controls the balance between coherence and diversity of the output. (default: 5)'
        )

        parser.add_argument(
            '--num_ctx',
            type=int,
            default=2048,
            help='Sets the size of the context window used to generate the next token. (default: 2048)'
        )

        parser.add_argument(
            '--num_gpu',
            type=int,
            default=None,
            help='The number of GPUs to use. On macOS it defaults to 1 to enable metal support, 0 to disable.'
        )

        parser.add_argument(
            '--num_thread',
            type=int,
            default=None,
            help='Sets the number of threads to use during computation.'
        )

        parser.add_argument(
            '--num_predict',
            type=int,
            default=128,
            help='Maximum number of tokens to predict when generating text. (default: 128, -1 = infinite generation, -2 = fill context)'
        )

        parser.add_argument(
            '--repeat_last_n',
            type=int,
            default=64,
            help='Sets how far back for the model to look back to prevent repetition. (default: 64, 0 = disabled, -1 = num_ctx)'
        )

        parser.add_argument(
            '--repeat_penalty',
            type=float,
            default=1.1,
            help='Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (default: 1.1)'
        )

        parser.add_argument(
            '--temperature',
            type=float,
            default=0.8,
            help='The temperature of the model. Increasing the temperature will make the model answer more creatively. (default: 0.8)'
        )

        parser.add_argument(
            '--stop',
            type=str,
            nargs='+',
            default=None,
            help='Sets the stop tokens to use.'
        )

        parser.add_argument(
            '--tfs_z',
            type=float,
            default=1.0,
            help='Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)'
        )

        parser.add_argument(
            '--top_k',
            type=int,
            default=40,
            help='Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (default: 40)'
        )

        parser.add_argument(
            '--top_p',
            type=float,
            default=0.9,
            help='Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (default: 0.9)'
        )

        parser.add_argument(
            '--system',
            type=str,
            default=None,
            help='System prompt (overrides what is defined in the Modelfile)'
        )

        parser.add_argument(
            '--template',
            type=str,
            default=None,
            help='Full prompt or prompt template (overrides what is defined in the Modelfile)'
        )

        parser.add_argument(
            '--format',
            type=str,
            default=None,
            help='Specify the format of the output (e.g., json)'
        )

        parser.add_argument(
            '--timeout',
            type=int,
            default=None,
            help='Timeout for the request stream'
        )

        parser.add_argument(
            '--keep_alive',
            type=Union[int, str],
            default=None,
            help='How long the model will stay loaded into memory.'
        )

        parser.set_defaults(llm=lambda args: get_ollama(**args))

    @staticmethod
    def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
        parser.print_help()
