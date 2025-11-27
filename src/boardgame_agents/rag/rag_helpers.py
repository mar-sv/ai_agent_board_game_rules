from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
load_dotenv()


def extend_chathistory(chat_history, user_input, llm_answer):
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=llm_answer)
    ])
    return chat_history


def get_retriever(n_search_kwargs=5):
    PG_DSN = os.getenv("DB_DSN")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = PGVector(
        connection=PG_DSN,
        embeddings=embeddings,
        collection_name="chunks"
    )

    return vectorstore.as_retriever(search_kwargs={"k": n_search_kwargs})


class Reranker:
    """Second-stage reranker using a cross-encoder."""

    def __init__(self, retriever, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5):
        self.retriever = retriever
        self.top_k = top_k
        self.model = CrossEncoder(model_name)

    def get_relevant_documents(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.top_k]]


def get_reranked_retriever(initial_k=5, final_k=2):
    base = get_retriever(k=initial_k)
    return Reranker(base, top_k=final_k)
