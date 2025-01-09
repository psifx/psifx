"""large language model tool."""

import re
from pathlib import Path
from typing import Union, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable

from psifx.io.txt import TxtReader
from psifx.io.yaml import YAMLReader
from psifx.text.llm.openai.tool import get_openai
from psifx.text.llm.ollama.tool import get_ollama
from psifx.text.llm.hf.tool import get_lc_hf
from psifx.text.llm.anthropic.tool import get_anthropic
from psifx.text.tool import TextTool


class LLMTool(TextTool):
    parsers: dict
    providers: dict

    def __init__(self, device: Optional[str] = '?', overwrite: Optional[bool] = False,
                 verbose: Optional[Union[bool, int]] = True):
        super().__init__(
            device,
            overwrite,
            verbose,
        )
        self.parsers = {
            'default': self.default_parser
        }
        self.providers = {
            'hf': get_lc_hf,
            'ollama': get_ollama,
            'openai': get_openai,
            'anthropic': get_anthropic
        }

    def chain_from_yaml(self, llm: BaseChatModel, yaml_path: Union[str, Path]) -> RunnableSerializable:
        """
        Return a chain from a yaml file.

        :param llm: A large language model.
        :param yaml_path: Path to the .yaml config file.
        :return: A chain.
        """
        dictionary = YAMLReader.read(yaml_path)
        prompt = LLMTool.load_template(prompt=dictionary['prompt'])
        parser = self.instantiate_parser(**dictionary.get('parser', {'kind': 'default'}))
        return LLMTool.make_chain(llm=llm, prompt=prompt, parser=parser)

    def chains_from_yaml(self, llm: BaseChatModel, yaml_path: Union[str, Path]) -> dict[str:RunnableSerializable]:
        """
        Return a dictionary of chains from a yaml file.

        :param llm: A large language model.
        :param yaml_path: Path to the .yaml config file.
        :return: A dictionary of strings mapped to chains.
        """

        def make_chain_wrapper(prompt, parser=None):
            if parser is None:
                parser = {'kind': 'default'}
            prompt = LLMTool.load_template(prompt=prompt)
            parser = self.instantiate_parser(**parser)
            return LLMTool.make_chain(llm=llm, prompt=prompt, parser=parser)

        return {key: make_chain_wrapper(**value) for key, value in YAMLReader.read(yaml_path).items()}

    @staticmethod
    def make_chain(llm: BaseChatModel, prompt: ChatPromptTemplate,
                   parser: Callable) -> RunnableSerializable:
        """
        Return a chain composed of the prompt, llm, and parser.

        :param llm: A large language model.
        :param prompt: A prompt template.
        :param parser: A parser in the form of a chain.
        :return: A chain composed of the prompt, llm, and parser.
        """

        def dict_wrapper(function):
            return lambda dictionary: function(**dictionary)

        return (RunnableParallel({'data': RunnablePassthrough(), 'generation': prompt | llm}) |
                dict_wrapper(parser))

    @staticmethod
    def load_template(prompt: Union[str, Path]) -> ChatPromptTemplate:
        """
        Return a chat prompt template from a string or a path to a .txt file.

        :param prompt: String or path to a .txt file.
        :return: A chat prompt template.
        """
        try:
            prompt = TxtReader.read(path=prompt)
        except NameError:
            pass
        pattern = r"(user|human|assistant|ai|system):\s(.*?)(?=user:|human:|assistant:|ai:|system:|$)"
        matches = re.findall(pattern, prompt, re.DOTALL)
        matches = [(role, msg.strip()) for role, msg in matches]
        if prompt and not matches:
            print("Template was not parsed correctly, please specify roles")
        return ChatPromptTemplate.from_messages(matches)

    def instantiate_llm(self, provider: str, **provider_kwargs) -> BaseChatModel:
        """
        Return a large language model from a provider and key value arguments.

        :param provider: Name of the llm provider.
        :param provider_kwargs: Key value arguments for instantiating the llm.
        :return: A large language model.
        """
        if self.verbose:
            print('\n' + '-' * 20 + '\nModel configuration')
            print(f"provider: {provider}")
            for key, value in provider_kwargs.items():
                print(f"{key}: {value}")
            print('-' * 20)

        if provider in self.providers:
            return self.providers[provider](**provider_kwargs)
        else:
            valid_providers = ', '.join(self.providers.keys())
            raise NameError(f'model provider should be one of: {valid_providers}')

    def instantiate_parser(self, kind: str = 'default', **parser_kwargs) -> Callable:
        """
        Return a kind of parser with key value arguments.

        :param kind: Name of the kind of parser.
        :param parser_kwargs: Key value arguments for the parser.
        :return: A kind of parser with key value arguments.
        """
        if kind in self.parsers:
            parser = self.parsers[kind]
            return lambda generation, data: parser(generation=generation,
                                                   data=data,
                                                   **parser_kwargs)
        else:
            valid_parsers = ', '.join(self.parsers.keys())
            raise NameError(f'parser kind should be one of: {valid_parsers}')

    def default_parser(self, generation: AIMessage, data: dict,
                       start_after: Optional[str] = None,
                       to_lower: Optional[bool] = False,
                       expect: Optional[list[str]] = None) -> str:
        """
        Parse a message starting from start_flag and check whether it is one of the expected_labels.

        :param generation: Message from a llm.
        :param data: Additional data from the chain.
        :param start_after: If not None, parse the message from the last instance of start_after.
        :param to_lower: If True, change the output to lowercase (it applies subsequently to start_after).
        :param expect: If not None, when the final output is not one of the expected labels prints an error message.
        :return: The parsed message.
        """
        output = generation.content
        if start_after:
            parts = output.split(start_after)
            if len(parts) == 1:
                print(f"START FLAG NOT PRESENT IN GENERATION\nDATA:\n{data}\nGENERATION:\n{generation.content}")
            output = parts[-1]
        if to_lower:
            output = output.lower()
        output = output.strip()

        str_data = f"\nDATA:\n{data}"
        str_generation = f"\nGENERATION:\n{generation.content}"
        str_output = f"\nOUTPUT:\n{output}"

        if expect and output not in expect:
            print(f"UNEXPECTED OUTPUT{str_data}{str_generation}{str_output}")
        elif self.verbose:
            if generation.content != output:
                print(f"{str_data}{str_generation}{str_output}")
            else:
                print(f"{str_data}{str_output}")
        return output
