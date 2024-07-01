from langchain_community.chat_models import ChatOllama


class WrapperChatOllama(ChatOllama):
    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except Exception as e:
            print("Make sure you downloaded the model from Ollama")
            print(e)
            raise e


def get_ollama(model='llama3', **kwargs):
    if model not in ([available_model['name'] for available_model in ollama.list()['models']]):
        pull_model(model)

    return ChatOllama(model=model, **kwargs)


def pull_model(model):
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
