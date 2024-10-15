"""large language model command-line interface."""

import argparse

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable

from psifx.io.yaml import YAMLReader
from psifx.text.llm.tool import LLMTool


def instantiate_llm_tool(args: argparse.Namespace) -> LLMTool:
    """
    Instantiate a LLMTool.
    :param args: Argument Namespace.
    :return: A LLMTool.
    """
    return LLMTool(
        device=getattr(args, 'device', '?'),
        overwrite=getattr(args, 'overwrite', False),
        verbose=getattr(args, 'verbose', True)
    )


def instantiate_llm(args: argparse.Namespace, llm_tool: LLMTool) -> BaseChatModel:
    """
    Instantiate a large language model.
    :param args:  Argument Namespace.
    :param llm_tool: A LLMTool.
    :return: A large language model.
    """
    llm_kw_args = args.model_config or {}
    for key in ('api_key', 'provider', 'model'):
        assert key not in llm_kw_args, f"{key} should not be specified in the model config file, please provide it with --{key} instead"
    llm_kw_args['provider'] = args.provider
    llm_kw_args['model'] = args.model
    if args.api_key is not None:
        llm_kw_args['api_key'] = args.api_key
    return llm_tool.instantiate_llm(**llm_kw_args)


def add_llm_argument(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for large language model.
    :param parser: An argument parser.
    """
    parser.add_argument(
        '--provider',
        type=str,
        default='ollama',
        choices=['ollama', 'hf', 'openai', 'anthropic'],
        required=False,
        help="The large language model provider. Choices are 'ollama', 'hf', 'openai', or 'anthropic'. Default is 'ollama'."
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        required=False,
        help="The large language model to use. This depends on the provider."
    )
    parser.add_argument(
        '--model_config',
        type=YAMLReader.read,
        required=False,
        help="A model .yaml configuration file."
    )
    parser.add_argument(
        '--api_key',
        type=str,
        required=False,
        help="Corresponding API key for 'hf', 'openai', or 'anthropic'."
    )
    parser.set_defaults(instantiate_llm_tool=instantiate_llm_tool)
    parser.set_defaults(instantiate_llm=instantiate_llm)


def instantiate_chain(args: argparse.Namespace, llm_tool: LLMTool, llm: BaseChatModel) -> RunnableSerializable:
    """
    Instantiate a langchain chain.
    :param args: Argument Namespace.
    :param llm_tool: A LLMTool.
    :param llm: A large language model.
    :return: A langchain chain.
    """
    dictionary = YAMLReader.read(args.yaml_path)
    prompt = llm_tool.load_template(prompt=dictionary['prompt'])
    parser = llm_tool.instantiate_parser(**dictionary.get('parser', {'kind': 'default'}))
    return llm_tool.make_chain(llm=llm, prompt=prompt, parser=parser)


def add_llm_instruction_argument(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for llm with instruction.
    :param parser: An argument parser.
    """
    add_llm_argument(parser)
    parser.add_argument(
        '--instruction',
        type=str,
        required=True,
        help="path to a .yaml file containing the prompt and parser")

    parser.set_defaults(instantiate_chain=instantiate_chain)
