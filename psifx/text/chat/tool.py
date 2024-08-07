from psifx.text.llm.tool import LLMUtility
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import MessagesPlaceholder
from psifx.tool import Tool
from psifx.io.txt import TxtWriter


class ChatTool(Tool):
    """
    Base class for chat tools.
    """

    def __init__(self, llm, **kwargs):
        super().__init__(device="?", **kwargs)
        self.llm = llm

    def chat(self, prompt: str, save_file: str):
        if save_file:
            TxtWriter.check(save_file, self.overwrite)
        prompt_template = LLMUtility.load_template(prompt)
        prompt_template.append(MessagesPlaceholder(variable_name="messages"))

        chat_history = ChatMessageHistory()
        chain = prompt_template | self.llm
        while (reply := input('User: ')) != 'exit':
            chat_history.add_user_message(reply)
            ai_message: AIMessage = chain.invoke({"messages": chat_history.messages})
            chat_history.add_ai_message(ai_message)
            print(f'Chatbot: {ai_message.content}')
            if save_file:
                TxtWriter.write(
                    content='\n'.join((f"{message.type}: {message.content}" for message in chat_history.messages)),
                    path=save_file,
                    overwrite=True)
