import re
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from psifx.io.txt import TxtReader
from psifx.io.yaml import YAMLReader
from psifx.text.llm.openai.tool import get_openai
from psifx.text.llm.ollama.tool import get_ollama
from psifx.text.llm.hf.tool import get_lc_hf
from psifx.text.llm.anthropic.tool import get_anthropic


class LLMTool:
    parsers: dict
    providers: dict

    def __init__(self):
        self.parsers = {
            'default': self.default_parser
        }
        self.providers = {
            'hf': get_lc_hf,
            'ollama': get_ollama,
            'openai': get_openai,
            'anthropic': get_anthropic
        }

    def llm_from_yaml(self, yaml_path):
        data = YAMLReader.read(yaml_path)
        assert 'provider' in data, 'Please give a provider'
        return self.instantiate_llm(**data)

    def chains_from_yaml(self, llm, yaml_path):
        def make_chain_wrapper(prompt, parser):
            prompt = LLMTool.load_template(prompt=prompt)
            parser = self.instantiate_parser(**parser)
            return LLMTool.make_chain(llm=llm, prompt=prompt, parser=parser)

        return {key: make_chain_wrapper(**value) for key, value in YAMLReader.read(yaml_path).items()}

    @staticmethod
    def make_chain(llm, prompt: ChatPromptTemplate, parser):
        def dict_wrapper(function):
            return lambda dictionary: function(**dictionary)

        return (RunnableParallel({'data': RunnablePassthrough(), 'generation': prompt | llm}) |
                dict_wrapper(parser))

    @staticmethod
    def load_template(prompt: str) -> ChatPromptTemplate:
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

    def instantiate_llm(self, provider, **kwargs):
        if provider in self.providers:
            return self.providers[provider](**kwargs)
        else:
            valid_providers = ', '.join(self.providers.keys())
            raise NameError(f'model provider should be one of: {valid_providers}')

    def instantiate_parser(self, kind, **kwargs):
        if kind in self.parsers:
            parser = self.parsers[kind]
            return lambda generation, data: parser(generation=generation,
                                                   data=data,
                                                   **kwargs)
        else:
            valid_parsers = ', '.join(self.providers.keys())
            raise NameError(f'parser kind should be one of: {valid_parsers}')

    @staticmethod
    def default_parser(generation: AIMessage, data: dict, start_flag: str = None, expected_labels: list[str] = None,
                       verbose=False) -> str:
        answer = generation.content
        if start_flag:
            answer = answer.split(start_flag)[-1]
        if expected_labels and answer not in expected_labels:
            print(f"PROBLEMATIC GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {answer}")
        elif verbose:
            print(f"WELL PARSED GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {answer}")
        return answer
