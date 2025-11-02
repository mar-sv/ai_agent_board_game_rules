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
from datetime import datetime
from typing import List, Tuple
import psycopg
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
PG_DSN = os.getenv("PG_DSN", "postgresql://user:pass@localhost:5432/ragdb")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))
# ---------------------------

# Fast SHA1 of full text for idempotency/versioning


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def read_pdf(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        # keep figure/table captions close to page content
        caps = [ln for ln in t.splitlines() if re.match(
            r"^(Figure|Table)\s+\d+[:\.]", ln)]
        pages.append(
            (i+1, (t + ("\n" + "\n".join(caps) if caps else "")).strip()))
    full = "\n\n".join(txt for _, txt in pages)
    return full, pages


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


def guess_section(chunk_text: str) -> str:
    for ln in chunk_text.split("\n")[:6]:
        ln = ln.strip()
        if re.match(r"^\d+(\.\d+)*\s+\S+", ln) or re.match(r"^[A-Z][A-Za-z0-9\s\-]{3,}$", ln):
            return ln[:120]
    return "Body"


DDL = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY,
  doc_title TEXT,
  source_path TEXT,
  version TEXT,
  content_sha1 TEXT UNIQUE,
  pages INT,
  ingested_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT,
  text TEXT,
  embedding vector(384),
  word_count INT,
  page_start INT,
  page_end INT,
  section_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
-- Optional sparse search for hybrid retrieval
DO $$ BEGIN
  ALTER TABLE chunks ADD COLUMN ts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(text,''))) STORED;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;
CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_ts    ON chunks USING GIN (ts);
-- pgvector IVF index (requires ANALYZE; tune lists per data size)
CREATE INDEX IF NOT EXISTS idx_chunks_emb ON chunks USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
"""

UPSERT_DOC = """
INSERT INTO documents (doc_id, doc_title, source_path, version, content_sha1, pages)
VALUES (%s,%s,%s,%s,%s,%s)
ON CONFLICT (doc_id) DO NOTHING
"""

UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, embedding, word_count, page_start, page_end, section_path)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
            doc_id = content_id  # stable id for dedupe/versioning
            version = datetime.utcnow().strftime("%Y%m%d")

            with conn.cursor() as cur:
                cur.execute(UPSERT_DOC, (doc_id, title, os.path.abspath(
                    path), version, content_id, len(pages)))

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
                        "page_end": page_no,
                        "section_path": guess_section(chunk)
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
                            r["section_path"],
                        ),
                    )
            print(f"[ok] Ingested {title}: {len(chunk_rows)} chunks")

    print("Done. TIP: run ANALYZE; and consider increasing ivfflat lists as the corpus grows.")


if __name__ == "__main__":
    main(sys.argv[1:])
