"""ollama model."""

from langchain_ollama import ChatOllama
import ollama
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm
import os
import time
import subprocess

def get_ollama(model: str = 'llama3.1', **kwargs) -> BaseChatModel:
    """
    Get an ollama langchain base chat model.

    :param model: Name of the model.
    :param kwargs: Key value arguments to pass to the model.
    :return: An ollama langchain base chat model.
    """
    try:
        ollama.list()
    except Exception as e:
        print(e)
        print("Try starting ollama serve")
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        time_limit = 10
        start_time = time.time()
        while time.time() - start_time < time_limit:
            stderr_line = process.stderr.readline().decode().strip()
            if "Listening on" in stderr_line:
                print(f"Service started successfully")
                break
    if model not in ([available_model['model'] for available_model in ollama.list()['models']]):
        pull_model(model)

    return ChatOllama(model=model, **kwargs)


def pull_model(model: str):
    """
    Pull a model from ollama.

    :param model: Name of the model.
    """
    print(f"pulling {model}")
    pbar = None
    current_digest = None
    for data in ollama.pull(model, stream=True):
        if data.get('digest') is None:
            continue
        if data['digest'] != current_digest:
            if pbar:
                pbar.close()
            current_digest = data['digest']
            total_size = data['total']
            pbar = tqdm(total=total_size, desc=f"{data['status']}", unit="B", unit_scale=True)

        completed_size = data.get('completed', 0)
        pbar.n = completed_size
        pbar.refresh()

    if pbar:
        pbar.close()
    print(f"successfully pulled {model}")
