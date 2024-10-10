"""openai model."""

import os
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_openai(model: str = 'gpt-4o', api_key: Optional[str] = None, temperature: int = 0,
               max_tokens: Optional[int] = None, timeout: Optional[int] = None,
               max_retries: int = 2, **kwargs) -> BaseChatModel:
    """
    Get an openai langchain base chat model.

    :param model: Name of the model to use.
    :param api_key: Api key (if not provided the user is prompted for it).
    :param temperature: Temperature.
    :param max_tokens: Max number of tokens.
    :param timeout: Timeout.
    :param max_retries: Max retries.
    :param kwargs: Optional key value arguments.
    :return: An openai langchain base chat model.
    """
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )
