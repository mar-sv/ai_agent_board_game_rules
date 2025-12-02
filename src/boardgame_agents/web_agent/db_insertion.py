import os
from pathlib import Path
import uuid
import hashlib
import re
from typing import List


from langchain_community.document_loaders import PDFPlumberLoader
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import psycopg2

load_env = load_dotenv()


PG_DSN = os.getenv("DB_DSN", "")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))


def process_and_insert_pdf(pdf_path: str, creator: str):
    doc_name = Path(pdf_path).stem

    loader = PDFPlumberLoader(pdf_path)
    pages = loader.load()

    for d in pages:
        d.metadata = {
            **(d.metadata or {}),
            "document_name": doc_name,
            "source": doc_name,
            "page": d.metadata.get("page", None),
            "creator": creator
        }
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vs = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="chunks",
        connection=PG_DSN
    )


def document_exists_sql(doc_name: str) -> bool:
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()

    cur.execute("""
        SELECT 1 FROM langchain_pg_embedding
        WHERE cmetadata->>'document_name' = %s
        LIMIT 1;
    """, (doc_name,))

    exists = cur.fetchone() is not None

    cur.close()
    conn.close()
    return exists


if __name__ == "__main__":
    paths = ["pdfs/Terraforming Mars.pdf"]
    process_and_insert_pdf(paths)
    # main(paths)
