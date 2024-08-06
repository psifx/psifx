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


class LLMUtility:
    @staticmethod
    def chains_from_yaml(llm, yaml):
        def make_chain_wrapper(prompt, parser):
            prompt = LLMUtility.load_template(prompt=prompt)
            parser = LLMUtility.instantiate_parser(**parser)
            return LLMUtility.make_chain(llm=llm, prompt=prompt, parser=parser)

        return {key: make_chain_wrapper(**value) for key, value in YAMLReader.read(yaml).items()}

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
        pattern = r"(user|assistant|system):\s(.*?)(?=user:|assistant:|system:|$)"
        matches = re.findall(pattern, prompt, re.DOTALL)
        matches = [(role, msg.strip()) for role, msg in matches]
        if prompt and not matches:
            print("Template was not parsed correctly, please specify roles")
        return ChatPromptTemplate.from_messages(matches)

    @staticmethod
    def llm_from_yaml(yaml_config):
        data = YAMLReader.read(yaml_config)
        assert 'provider' in data, 'Please give a provider'
        return LLMUtility.instantiate_llm(**data)

    @staticmethod
    def instantiate_llm(provider, **kwargs):
        provider_map = {
            'hf': get_lc_hf,
            'ollama': get_ollama,
            'openai': get_openai,
            'anthropic': get_anthropic
        }

        if provider in provider_map:
            return provider_map[provider](**kwargs)
        else:
            valid_providers = ', '.join(provider_map.keys())
            raise NameError(f'provider should be one of: {valid_providers}')

    @staticmethod
    def instantiate_parser(kind, **kwargs):
        if kind == 'split':
            parser = LLMUtility.split_parser
        elif kind == 'segment':
            parser = LLMUtility.segment_parser
        elif kind == 'default':
            parser = LLMUtility.default_parser
        else:
            raise NameError
        return lambda generation, data: parser(generation=generation,
                                               data=data,
                                               **kwargs)

    @staticmethod
    def segment_parser(generation: AIMessage, data: dict, start_flag: str, left_separator: str, right_separator: str,
                       text_to_segment='text_to_segment', verbose=False) -> list[str]:
        answer = generation.content
        if start_flag:
            answer = answer.split(start_flag)[-1]
        segments = answer.replace(right_separator, "").split(left_separator)
        segments = [segment.strip() for segment in segments]

        reconstruction = []

        remaining_message = data[text_to_segment]

        for segment in segments[:-1]:
            match = re.search(re.escape(segment), remaining_message)
            if match:
                reconstruction.append(remaining_message[:match.end()].strip())
                remaining_message = remaining_message[match.end():]

        if remaining_message.strip():
            reconstruction.append(remaining_message.strip())

        if reconstruction != segments:
            print(
                f"PROBLEMATIC GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {reconstruction}")
        elif verbose:
            print(f"WELL PARSED GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {reconstruction}")
        return reconstruction

    @staticmethod
    def split_parser(generation: AIMessage, data: dict, start_flag: str, separator: str,
                     text_to_segment='text_to_segment', verbose=False) -> list[str]:
        answer = generation.content
        if start_flag:
            answer = answer.split(start_flag)[-1]
        segments = answer.split(separator)
        segments = [segment.strip() for segment in segments]

        reconstruction = []

        remaining_message = data[text_to_segment]

        for segment in segments[:-1]:
            if segment:
                match = re.search(re.escape(segment), remaining_message)
                if match:
                    reconstruction.append(remaining_message[:match.end()].strip())
                    remaining_message = remaining_message[match.end():]

        if remaining_message.strip():
            reconstruction.append(remaining_message.strip())

        if reconstruction != segments:
            print(
                f"PROBLEMATIC GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {reconstruction}")
        elif verbose:
            print(f"WELL PARSED GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {reconstruction}")
        return reconstruction

    @staticmethod
    def default_parser(generation: AIMessage, data: dict, start_flag: str=None, expected_labels: list[str] = None,
                       verbose=False) -> str:
        answer = generation.content
        if start_flag:
            answer = answer.split(start_flag)[-1]
        if expected_labels and answer not in expected_labels:
            print(f"PROBLEMATIC GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {answer}")
        elif verbose:
            print(f"WELL PARSED GENERATION: {generation.content}\nDATA: {data}\nPARSED AS: {answer}")
        return answer
