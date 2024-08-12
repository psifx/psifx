import os
import getpass
from langchain_anthropic import ChatAnthropic

def get_anthropic(model='claude-3-5-sonnet-20240620', api_key=None, temperature=0, max_tokens=None, timeout=None,
                  max_retries=2, **kwargs):
    api_key = api_key or getpass.getpass("Enter your Anthropic API key: ")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )
