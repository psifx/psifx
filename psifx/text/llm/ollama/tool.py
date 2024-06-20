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
    return WrapperChatOllama(model=model, **kwargs)
