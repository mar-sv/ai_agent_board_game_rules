from ragas.testset import Testset  # just for type hint / clarity
from sentence_transformers import CrossEncoder
from langchain.schema import Runnable
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
import os
from typing import List

from datasets import Dataset

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# <-- your existing code pieces ---------------------------------------------

EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


PG_DSN = os.getenv("DB_DSN")


def generate_llm(temperature=0):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def get_retriever(k: int = 5):
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


def get_reranked_retriever(initial_k: int = 8, final_k: int = 4) -> Reranker:
    base = get_retriever(k=initial_k)
    return Reranker(base, top_k=final_k)


# --------------------------------------------------------------------
# 1) Assume you ALREADY generated a Ragas testset:
#    testset = generator.generate_with_langchain_docs(docs, testset_size=20)
#    Here we just take it as given.
# --------------------------------------------------------------------


def build_eval_dataset_from_testset(testset: "Testset") -> Dataset:
    """
    Take the Ragas-generated testset, run YOUR retriever on each question,
    and build a HF Dataset with the columns Ragas expects for retrieval metrics:
      - user_input
      - retrieved_contexts
      - reference
    """
    hf_test_ds = testset.to_dataset()  # has user_input, reference, reference_contexts, ...

    retriever = get_reranked_retriever()

    rows = []
    for row in hf_test_ds:
        question = row["user_input"]
        ref_answer = row["reference"]  # ground-truth answer

        # Run your real retriever (with reranker)
        docs = retriever.invoke(question)
        retrieved_contexts = [d.page_content for d in docs]

        rows.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "reference": ref_answer,
            }
        )

    return Dataset.from_list(rows)


def main(testset: "Testset"):
    # 1) Build eval dataset with retrieved contexts
    eval_ds = build_eval_dataset_from_testset(testset)

    # 2) Run Ragas retrieval metrics.
    #    context_precision + context_recall use:
    #      user_input, retrieved_contexts, reference :contentReference[oaicite:1]{index=1}
    results = evaluate(
        eval_ds,
        metrics=[context_precision, context_recall],
        # let Ragas wrap your LangChain LLM + embeddings for judging
        llm=generate_llm(),
        embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    )

    # 3) Inspect scores
    df = results.to_pandas()
    print(df)
    print("Mean context_precision:", df["context_precision"].mean())
    print("Mean context_recall:", df["context_recall"].mean())


# Example usage:
# from your_generation_script import testset
# main(testset)
