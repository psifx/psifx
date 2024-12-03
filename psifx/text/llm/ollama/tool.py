from langchain_community.chat_models import ChatOllama
def get_ollama(model='llama3', **kwargs):
    return ChatOllama(model=model, **kwargs)
