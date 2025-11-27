from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def extend_chathistory(chat_history, user_input, llm_answer):
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=llm_answer)
    ])
    return chat_history


def get_retriver(n_search_kwargs=5):
    PG_DSN = os.getenv("DB_DSN")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = PGVector(
        connection=PG_DSN,
        embeddings=embeddings,
        collection_name="chunks"
    )

    return vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})
