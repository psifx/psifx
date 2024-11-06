"""anthropic model."""

import os
from typing import Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


def get_anthropic(model: str = 'claude-3-5-sonnet-20240620', api_key: Optional[str] = None, temperature: int = 0,
                  max_tokens: Optional[int] = None, timeout: Optional[int] = None,
                  max_retries: int = 2, **kwargs) -> BaseChatModel:
    """
    Get an anthropic langchain base chat model.

    :param model: Name of the model to use.
    :param api_key: Api key (if not provided the user is prompted for it).
    :param temperature: Temperature.
    :param max_tokens: Max number of tokens.
    :param timeout: Timeout.
    :param max_retries: Max retries.
    :param kwargs: Optional key value arguments.
    :return: An anthropic langchain base chat model.
    """
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )
