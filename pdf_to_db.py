"""
pip install psycopg[binary] pypdf sentence-transformers

Usage:
  python ingest_pdfs.py ./docs/file1.pdf ./docs/file2.pdf
  # or: python ingest_pdfs.py ./docs/*.pdf

Make sure pgvector is installed on your DB:
  CREATE EXTENSION IF NOT EXISTS vector;
"""

import os
import sys
import uuid
import hashlib
import re
from typing import List, Tuple
import psycopg
from sentence_transformers import SentenceTransformer
import pdfplumber
from dotenv import load_dotenv

load_env = load_dotenv()

# ---------- CONFIG ----------
# connection_string = "postgresql://{user}:{password}@{host}:{port}/{db_name}"
PG_DSN = os.getenv("DB_DSN", "")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))
# ---------------------------

# Fast SHA1 of full text for idempotency/versioning


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def read_pdf(pdf_path: str):
    pages = []
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:

        for page_n, page in enumerate(pdf.pages):
            page_text = page.extract_text().replace("\n", " ")
            pages.append((page_n + 1, page_text))
            full_text += page_text

        return full_text, pages


def to_paragraphs(page_text: str) -> List[str]:
    paras = [re.sub(r"\s+", " ", p).strip()
             for p in re.split(r"\n\s*\n", page_text)]
    return [p for p in paras if len(p.split()) > 5]


def chunk_words(paras: List[str], max_words=220, overlap=60) -> List[str]:
    chunks, buf = [], []
    wcount = 0
    for para in paras:
        words = para.split()
        if wcount + len(words) > max_words and buf:
            chunk = " ".join(buf)
            chunks.append(chunk)
            # overlap tail
            tail = " ".join(chunk.split()[-overlap:]) if overlap else ""
            buf = [tail] if tail else []
            wcount = len(tail.split()) if tail else 0
        buf.append(para)
        wcount += len(words)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


DDL = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY,
  doc_title TEXT,
  content_sha1 TEXT UNIQUE,
  pages INT
);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT,
  text TEXT,
  embedding vector(384),
  word_count INT,
  page_start INT,
  page_end INT
);
"""

UPSERT_DOC = """
INSERT INTO documents (doc_id, doc_title, content_sha1, pages)
VALUES (%s,%s,%s,%s)
ON CONFLICT (doc_id) DO NOTHING
"""

UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, embedding, word_count, page_start, page_end)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (chunk_id) DO NOTHING
"""


def main(paths: List[str]):
    if not paths:
        print("No input PDFs provided.")
        return
    model = SentenceTransformer(EMBED_MODEL)
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

        for path in paths:
            title = os.path.splitext(os.path.basename(path))[0]
            full, pages = read_pdf(path)
            content_id = sha1(full)
            doc_id = content_id

            with conn.cursor() as cur:
                cur.execute(UPSERT_DOC, (doc_id, title,
                            content_id, len(pages)))

            # Build page-aware chunks
            chunk_rows = []
            idx = 0
            for page_no, page_text in pages:
                paras = to_paragraphs(page_text)
                for chunk in chunk_words(paras, CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS):
                    idx += 1
                    chunk_rows.append({
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "text": chunk,
                        "word_count": len(chunk.split()),
                        "page_start": page_no,
                        "page_end": page_no
                    })

            if not chunk_rows:
                print(f"[warn] No text extracted from {path}")
                continue

            # Embed in batches
            B = 128
            texts = [r["text"] for r in chunk_rows]
            for i in range(0, len(texts), B):
                sub = texts[i:i+B]
                vecs = model.encode(sub, normalize_embeddings=True)
                for r, v in zip(chunk_rows[i:i+B], vecs):
                    r["embedding"] = list(map(float, v))

            # Upsert
            with conn.cursor() as cur:
                for r in chunk_rows:
                    cur.execute(
                        UPSERT_CHUNK,
                        (
                            r["chunk_id"],
                            r["doc_id"],
                            r["chunk_index"],
                            r["text"],
                            r["embedding"],
                            r["word_count"],
                            r["page_start"],
                            r["page_end"],
                        ),
                    )
            print(f"[ok] Ingested {title}: {len(chunk_rows)} chunks")


if __name__ == "__main__":
    paths = ["pdfs/catan.pdf"]
    main(paths)
