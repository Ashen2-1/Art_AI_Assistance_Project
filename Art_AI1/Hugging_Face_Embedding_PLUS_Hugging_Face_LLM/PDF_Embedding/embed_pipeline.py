################################################################
### Phase II - Embedding Pipeline
### Takes PDF chunks -> BGE embed -> PostgreSQL + pgvector
###
### Usage:
###   python embed_pipeline.py <file.pdf>           # printed mode
###   python embed_pipeline.py <file.pdf> cursive   # cursive mode
###   python embed_pipeline.py --stats              # show DB info
###
### As a module:
###   from embed_pipeline import ingest_pdf, get_db_stats
################################################################

import os
import sys
from pathlib import Path
from typing import List, Optional

# Windows console UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer

# pdf_pipeline.py and db.py are in the same folder
sys.path.insert(0, str(Path(__file__).parent))
from pdf_pipeline import process_pdf
from db import init_db, insert_chunks, list_sources, get_stats, delete_source

# ── Config ────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# ── Embedding model (loaded once, reused) ─────────────────────
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[Embed] Loading BGE model: {EMBEDDING_MODEL} ...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[Embed] Embedding model ready.")
    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    model = get_embed_model()
    vecs  = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return vecs.tolist()


# ── Main ingestion function ───────────────────────────────────
def ingest_pdf(
    pdf_path: str,
    ocr_mode: str      = "printed",
    chunk_size: int    = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Full Phase II pipeline for one PDF:
      pdf_pipeline (extract + chunk)
          -> BGE embedding
          -> PostgreSQL + pgvector storage

    Re-ingesting the same PDF replaces its old chunks.

    Returns number of chunks stored.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_name = Path(pdf_path).name
    print(f"\n[Embed] -- Ingesting: {source_name}  (ocr_mode={ocr_mode})")

    # -- Step 1: Extract + chunk ----------------------------------
    chunks = process_pdf(
        pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_mode=ocr_mode,
    )
    if not chunks:
        print("[Embed] No chunks extracted -- skipping.")
        return 0

    # -- Step 2: Embed --------------------------------------------
    print(f"[Embed] Embedding {len(chunks)} chunks ...")
    embeddings = embed_texts(chunks)

    # -- Step 3: Store in PostgreSQL ------------------------------
    init_db()   # ensures table exists
    n = insert_chunks(source_name, chunks, embeddings)

    stats = get_stats()
    print(f"[Embed] OK  {n} chunks stored  |  DB total: {stats['total_chunks']} chunks")
    return n


# ── Convenience wrappers (same API as old ChromaDB version) ───
def get_db_stats() -> dict:
    """Return total chunk count and list of ingested sources."""
    return get_stats()


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python embed_pipeline.py <file.pdf>         # printed OCR")
        print("  python embed_pipeline.py <file.pdf> cursive # cursive OCR")
        print("  python embed_pipeline.py --stats            # show DB info")
        sys.exit(1)

    if sys.argv[1] == "--stats":
        init_db()
        stats = get_db_stats()
        print("\n-- PostgreSQL Stats ----------------------------------------")
        print(f"  Total chunks : {stats['total_chunks']}")
        print(f"  Sources      : {stats['sources'] or 'none'}")
        print("------------------------------------------------------------")
        sys.exit(0)

    pdf_file = sys.argv[1]
    mode     = sys.argv[2] if len(sys.argv) > 2 else "printed"

    n     = ingest_pdf(pdf_file, ocr_mode=mode)
    stats = get_db_stats()

    print("\n" + "=" * 52)
    print(f"  Ingested : {n} chunks  from  {Path(pdf_file).name}")
    print(f"  DB total : {stats['total_chunks']} chunks")
    print(f"  Sources  : {stats['sources']}")
    print("=" * 52)
