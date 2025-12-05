from ragas.testset import Testset  # just for type hint / clarity
from langchain_openai import ChatOpenAI
from src.boardgame_agents.rag.rag_helpers import get_reranked_retriever
from src.boardgame_agents.evaluation.generate_eval_data import generate_testset
import os


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


def build_eval_dataset_from_testset(testset: "Testset"):

    retriever = get_reranked_retriever()

    rows = []
    for row in testset:
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


def evaluate_rag():
    eval_ds = build_eval_dataset_from_testset(generate_testset())

    results = evaluate(
        eval_ds,
        metrics=[context_precision, context_recall],
        llm=generate_llm(),
        embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    )

    df = results.to_pandas()
    print(df)
    print("Mean context_precision:", df["context_precision"].mean())
    print("Mean context_recall:", df["context_recall"].mean())


if __name__ == "__main__":
    evaluate_rag()
