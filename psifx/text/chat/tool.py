from psifx.text.llm.tool import LLMUtility
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import MessagesPlaceholder
from psifx.tool import Tool


class ChatTool(Tool):
    """
    Base class for chat tools.
    """

    def __init__(self, llm, **kwargs):
        super().__init__(device="?", **kwargs)
        self.llm = llm

    def chat(self, prompt):
        prompt_template = LLMUtility.load_template(prompt)
        prompt_template.append(MessagesPlaceholder(variable_name="messages"))

        demo_ephemeral_chat_history = ChatMessageHistory()
        chain = prompt_template | self.llm
        while (reply := input('User: ')) != 'exit':
            demo_ephemeral_chat_history.add_user_message(reply)
            ai_message: AIMessage = chain.invoke({"messages": demo_ephemeral_chat_history.messages})
            demo_ephemeral_chat_history.add_ai_message(ai_message)
            print(f'Chatbot: {ai_message.content}')
