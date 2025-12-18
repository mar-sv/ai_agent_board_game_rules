from dotenv import load_dotenv
import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI

load_dotenv()


def extend_chathistory(chat_history, user_input, llm_answer):
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=llm_answer),
    ])
    return chat_history


def get_retriever(k: int = 5):
    PG_DSN = os.getenv("DB_DSN")
    print(PG_DSN)
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = PGVector(
        connection=PG_DSN,
        embeddings=embeddings,
        collection_name="chunks",
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})


class Reranker(Runnable):
    """Reranker using a cross-encoder."""

    def __init__(
        self,
        retriever: Runnable,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):

        self.retriever = retriever
        self.top_k = top_k
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def invoke(self, query: str, config=None) -> List[Document]:
        docs: List[Document] = self.retriever.invoke(query, config=config)
        if not docs:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.top_k]]


def get_reranked_retriever(initial_k: int = 5, final_k: int = 2) -> Reranker:
    base = get_retriever(k=initial_k)
    return Reranker(base, top_k=final_k)


def get_llm_model(temperature=0):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )
