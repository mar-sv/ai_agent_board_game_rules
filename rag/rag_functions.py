from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate


def extend_chathistory(chat_history, question, response):
    chat_history.extend([HumanMessage(content=question)],
                        AIMessage(content=response))
    return chat_history
