import getpass
from langchain_openai import ChatOpenAI

def get_openai(model='gpt-4o', api_key=None, temperature=0, max_tokens=None, timeout=None, max_retries=2, **kwargs):
    api_key = api_key or getpass.getpass("Enter your OpenAI API key: ")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )
