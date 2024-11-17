"""chatbot tool."""

from pathlib import Path
from typing import Union, Optional

from langchain_core.language_models import BaseChatModel

from psifx.text.llm.tool import LLMTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import MessagesPlaceholder
from psifx.text.tool import TextTool
from psifx.io.txt import TxtWriter


class ChatTool(TextTool):
    """
    text chat tool.

    :param llm: The large language model to discuss with.
    :param overwrite: Whether to overwrite existing files, otherwise raise an error.
    :param verbose: Whether to execute the computation verbosely."""

    def __init__(self, llm: BaseChatModel, overwrite: Optional[bool] = False,
                 verbose: Optional[Union[bool, int]] = True):
        super().__init__(device="?",
                         overwrite=overwrite,
                         verbose=verbose)
        self.llm = llm

    def chat(self, prompt: Optional[Union[str, Path]] = "", save_path: Optional[Union[str, Path]] = ""):
        """
        Chat with a llm from a starting prompt, while saving the conversation.

        :param prompt: Prompt to start the conversation with.
        :param save_path: Path to the .txt save file.
        """
        if save_path:
            TxtWriter.check(save_path, self.overwrite)
        prompt_template = LLMTool.load_template(prompt)
        prompt_template.append(MessagesPlaceholder(variable_name="messages"))

        chat_history = ChatMessageHistory()
        chain = prompt_template | self.llm
        while (reply := input('User: ')) != 'exit':
            chat_history.add_user_message(reply)
            ai_message: AIMessage = chain.invoke({"messages": chat_history.messages})
            chat_history.add_ai_message(ai_message)
            print(f'Chatbot: {ai_message.content}')
            if save_path:
                TxtWriter.write(
                    content='\n'.join((f"{message.type}: {message.content}" for message in chat_history.messages)),
                    path=save_path,
                    overwrite=True)
