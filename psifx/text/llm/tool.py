import re
from typing import Union

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from psifx.io.txt import TxtReader
from psifx.tool import Tool
from langchain_community.llms import Ollama
from psifx.io.json import JSONReader
import json
from psifx.text.llm.ollama.tool import get_ollama
from psifx.text.llm.hf.tool import get_lc_hf
class LLMTool(Tool):
    """
    Base class for text tools.
    """

    def __init__(self, model,
                 overwrite: bool = False,
                 verbose: Union[bool, int] = True):

        super().__init__(
            device="cuda",
            overwrite=overwrite,
            verbose=verbose,
        )

        try:
            data = JSONReader.read(model)
        except NameError:
            try:
                data = json.loads(model)
            except json.JSONDecodeError:
                data = {'provider': 'ollama', 'model': model}
        assert 'provider' in data, 'Please give a provider'
        assert data['provider'] in ['hf', 'ollama'], 'provider should be hf, or ollama'
        if data['provider'] == 'hf':
            data.pop('provider')
            self.llm = get_lc_hf(**data)
        elif data['provider'] == 'ollama':
            data.pop('provider')
            self.llm = get_ollama(**data)

    @staticmethod
    def load_template(template):
        try:
            template = TxtReader.read(path=template)
        except NameError:
            pass
        return ChatPromptTemplate.from_template(template=template)

    @staticmethod
    def split_parser(generation: AIMessage, message: str, start_flag: str, separator: str):
        answer = generation.content.split(start_flag)[-1]
        segments = answer.split(separator)
        segments = [segment.strip() for segment in segments]

        reconstruction = []
        remaining_message = message

        for segment in segments[:-1]:
            if segment:
                match = re.search(re.escape(segment), remaining_message)
                if match:
                    reconstruction.append(match.group().strip())
                    remaining_message = remaining_message[match.end():]

        if remaining_message.strip():
            reconstruction.append(remaining_message.strip())

        if reconstruction != segments:
            print(
                f"PROBLEMATIC GENERATION: {generation.content}\nINPUT: {message}\nPARSED AS: {reconstruction}")
        return reconstruction
    @staticmethod
    def parser(generation: AIMessage, message: str, start_flag: str, expected_labels: list[str] = None):
        answer = generation.content.split(start_flag)[-1].strip().lower()
        if expected_labels and answer not in expected_labels:
            print(f"PROBLEMATIC GENERATION: {generation.content}\nINPUT: {message}\nPARSED AS: {answer}")
        return answer

    @staticmethod
    def _dict_wrapper(function):
        return lambda dictionary: function(**dictionary)