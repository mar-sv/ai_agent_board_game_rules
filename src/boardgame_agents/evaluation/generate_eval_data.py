import os
from typing import List

import psycopg2
import psycopg2.extras

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.transforms.splitters import HeadlineSplitter

from ragas.llms import llm_factory


PG_DSN = os.getenv("DB_DSN")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def generate_llm(temperature: float = 0.0):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def load_chunks_from_pg(collection_name: str = "chunks"):
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        """,
        (collection_name,),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    docs: List[Document] = []
    for row in rows:
        page_content = row["document"]
        metadata = row.get("cmetadata") or {}
        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


def build_ragas_generator():
    llm_wrapper = LangchainLLMWrapper(generate_llm())
    # llm_wrapper = llm_factory(model=os.getenv(
    #    "LLM_MODEL"), client=generate_llm())
    emb_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    return TestsetGenerator(llm=llm_wrapper, embedding_model=emb_wrapper)


def generate_testset():
    chunks = load_chunks_from_pg(collection_name="chunks")
    print(f"Loaded {len(chunks)} chunks from PGVector")

    generator = build_ragas_generator()
    transforms = [HeadlineSplitter()]  # keep only what you want

    testset = generator.generate_with_langchain_docs(
        documents=chunks,
        testset_size=20,
        transforms=transforms,
        #   raise_exceptions=False,
    )
    return testset.to_dataset()


if __name__ == "__main__":
    generate_testset()
