################################################################
### Database — PostgreSQL 16 + pgvector
### Creates tables and indexes on first run.
### All other modules import get_conn() and init_db() from here.
################################################################

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional

# ── Connection config (override via environment variables) ────
DB_CONFIG = {
    "host"    : os.getenv("PG_HOST", "localhost"),
    "port"    : int(os.getenv("PG_PORT", "5433")),
    "dbname"  : os.getenv("PG_DB",   "art_ai"),
    "user"    : os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "Shenglingl@0606"),
}

EMBEDDING_DIM = 384   # BAAI/bge-small-en-v1.5 output dimension


# ── Connection helper ─────────────────────────────────────────
def get_conn():
    """Open and return a new psycopg2 connection."""
    return psycopg2.connect(**DB_CONFIG)


# ── Schema setup ──────────────────────────────────────────────
def init_db():
    """
    Create the pdf_chunks table and indexes if they don't exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:

            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Main chunks table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS pdf_chunks (
                    id          SERIAL       PRIMARY KEY,
                    source      VARCHAR(500) NOT NULL,
                    chunk_index INTEGER      NOT NULL,
                    content     TEXT         NOT NULL,
                    embedding   vector({EMBEDDING_DIM}),
                    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Fast lookup by PDF filename
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_source
                ON pdf_chunks (source);
            """)

            # HNSW vector index — cosine similarity, works on empty tables
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON pdf_chunks USING hnsw (embedding vector_cosine_ops);
            """)

        conn.commit()
        print("[DB] PostgreSQL schema ready.")
    finally:
        conn.close()


# ── CRUD helpers ──────────────────────────────────────────────
def insert_chunks(
    source: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> int:
    """
    Insert a batch of chunks + embeddings for one PDF source.
    Replaces any existing rows for the same source first.
    Returns number of rows inserted.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Clean old data for this source
            cur.execute("DELETE FROM pdf_chunks WHERE source = %s;", (source,))

            # Bulk insert
            rows = [
                (source, i, chunk, embeddings[i])
                for i, chunk in enumerate(chunks)
            ]
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                """
                INSERT INTO pdf_chunks (source, chunk_index, content, embedding)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s::vector)",
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def search_chunks(
    query_embedding: List[float],
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """
    Cosine similarity search via pgvector (<=> operator).
    Returns top_k closest chunks as list of dicts.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_filter:
                cur.execute(
                    """
                    SELECT source, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM   pdf_chunks
                    WHERE  source = %s
                    ORDER  BY embedding <=> %s::vector
                    LIMIT  %s;
                    """,
                    (query_embedding, source_filter, query_embedding, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT source, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM   pdf_chunks
                    ORDER  BY embedding <=> %s::vector
                    LIMIT  %s;
                    """,
                    (query_embedding, query_embedding, top_k),
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def list_sources() -> List[str]:
    """Return all unique PDF filenames stored in the DB."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT source FROM pdf_chunks ORDER BY source;"
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_stats() -> dict:
    """Return total chunk count and list of ingested sources."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pdf_chunks;")
            total = cur.fetchone()[0]
        return {"total_chunks": total, "sources": list_sources()}
    finally:
        conn.close()


def delete_source(source_name: str) -> int:
    """Delete all chunks for a given PDF. Returns rows deleted."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM pdf_chunks WHERE source = %s;", (source_name,)
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print(f"Total chunks : {stats['total_chunks']}")
    print(f"Sources      : {stats['sources'] or 'none'}")
