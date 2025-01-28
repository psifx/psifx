"""hugging face model."""

import getpass
import os
from typing import List, Optional, Any
import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_core.language_models.chat_models import SimpleChatModel, BaseChatModel
from huggingface_hub.errors import GatedRepoError


class HFChat(SimpleChatModel):
    pipeline: Any = None

    def __init__(self, pipeline, **kwargs: Any):
        super().__init__(pipeline=pipeline, **kwargs)

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        messages_dicts = [self._to_chatml_format(m) for m in messages]
        return self.pipeline(messages_dicts)[0]['generated_text']

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}


def get_lc_hf(**kwargs) -> BaseChatModel:
    """
    Get a hugging face langchain base chat model.

    :param kwargs: Key value argument to pass on.
    :return: A hugging face langchain base chat model.
    """
    pipeline = get_transformers_pipeline(**kwargs)
    return HFChat(pipeline=pipeline)


def get_transformers_pipeline(model: str, api_key: Optional[str] = None, quantization: Optional[str] = None,
                              model_kwargs: Optional[dict] = None, max_new_tokens: Optional[int] = None,
                              pipeline_kwargs: Optional[dict] = None):
    """
        Create a text generation pipeline using a specified pre-trained language model.

        Parameters:
            model (str): The name or path of the pre-trained language model.
            api_key (str, optional): HuggingFace token for authentification. Default is None.
            quantization (str, optional): The quantization type to apply to the model. Options are '4bit', '8bit', or None. Default is None.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model during initialization. Default is None.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Default is None.
            pipeline_kwargs (dict, optional): Additional keyword arguments to pass to the pipeline during initialization. Default is None.

        Returns:
            `pipeline` object: A text generation pipeline initialized with the specified model, tokenizer, and pipeline arguments.
    """

    # Ensure model_kwargs and pipeline_kwargs are initialized if not provided
    model_kwargs = model_kwargs or {}
    pipeline_kwargs = pipeline_kwargs or {}

    # Check for Env variable
    api_key = api_key or os.environ.get('HF_TOKEN')

    # Validate quantization parameter
    assert quantization in ('4bit', '8bit', None), "The parameter quantization should be '4bit', '8bit', or None."

    # Configure quantization if specified
    if quantization == '4bit':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == '8bit':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, token=api_key)
        llm = AutoModelForCausalLM.from_pretrained(model, token=api_key, **model_kwargs)
    except EnvironmentError as env_err:
        if isinstance(env_err.__cause__, GatedRepoError):
            if api_key is None:
                api_key = getpass.getpass(
                    f"The model {model} requires special authorization.\nPlease provide an authorized HuggingFace token:")
                tokenizer = AutoTokenizer.from_pretrained(model, token=api_key)
                llm = AutoModelForCausalLM.from_pretrained(model, token=api_key, **model_kwargs)
            else:
                print(f"The model {model} requires special authorization.\nMake sure you have access to it.")
                raise env_err
        else:
            raise env_err
    except Exception as e:
        print(f"Error caused by initializing model '{model}'.")
        raise e

    # Configure pipeline
    pipeline_kwargs['max_new_tokens'] = max_new_tokens

    # Create and return the pipeline
    return pipeline("text-generation", model=llm, tokenizer=tokenizer, return_full_text=False, **pipeline_kwargs)
