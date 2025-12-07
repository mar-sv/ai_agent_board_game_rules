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
from langchain_core.documents import Document
import psycopg2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

load_env = load_dotenv()


PG_DSN = os.getenv("DB_DSN", "")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))


def extract_pages_with_numbers(pdf_path, doc_name, creator):
    docs = []

    for page_idx, layout in enumerate(extract_pages(pdf_path)):
        page_text = []

        for element in layout:
            if isinstance(element, LTTextContainer):
                if txt := element.get_text().strip():
                    page_text.append(txt)

        full_text = "\n".join(page_text)
        if not full_text.strip():
            continue

        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    "document_name": doc_name,
                    "source": doc_name,
                    "creator": creator,
                    "page": page_idx + 1,
                },
            )
        )
        print(full_text)

    return docs


def process_and_insert_pdf(pdf_path: str, creator: str):
    doc_name = Path(pdf_path).stem

    pages = extract_pages_with_numbers(pdf_path, doc_name, creator)

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

    print(f"Inserted {doc_name} by {creator} in the database")


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


def wipe_langchain_pg():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE langchain_pg_embedding CASCADE;")
    cur.execute("TRUNCATE TABLE langchain_pg_collection CASCADE;")

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    from langchain_core.documents import Document
    wipe_langchain_pg()

    pdf_path = r"C:\Github\ai_agent_board_game_rules\pdfs\Terraforming Mars.pdf"
    process_and_insert_pdf(pdf_path, creator="test")
    # main(paths)
