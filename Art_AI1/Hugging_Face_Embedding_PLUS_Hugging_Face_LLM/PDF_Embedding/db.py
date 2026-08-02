################################################################
### Database — PostgreSQL 16 + pgvector
### Creates tables and indexes on first run.
### All other modules import get_conn() and init_db() from here.
################################################################

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
# ── Connection config (override via environment variables) ────
DB_CONFIG = {
    "host"    : os.getenv("PG_HOST", "localhost"),
    "port"    : int(os.getenv("PG_PORT", "5433")),
    "dbname"  : os.getenv("PG_DB",   "art_ai"),
    "user"    : os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "Shenglingl@0606"),
}

EMBEDDING_DIM = 384   # Gemini Embedding output dimension


# ── Connection helper ─────────────────────────────────────────
def get_conn():
    """Open a PostgreSQL connection."""

    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password is missing. Configure DATABASE_URL or PG_PASS."
        )

    return psycopg2.connect(**DB_CONFIG)


# ── Schema setup ──────────────────────────────────────────────
def init_db():
    """
    Create the pdf_chunks_gemini table and indexes if they don't exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:

            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Main chunks table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS pdf_chunks_gemini (
                    id          SERIAL       PRIMARY KEY,
                    source      VARCHAR(500) NOT NULL,
                    chunk_index INTEGER      NOT NULL,
                    content     TEXT         NOT NULL,
                    embedding   vector({EMBEDDING_DIM}),
                    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Upgrade existing databases without deleting existing rows.
            cur.execute("""
                ALTER TABLE pdf_chunks_gemini
                ADD COLUMN IF NOT EXISTS user_id TEXT
                NOT NULL DEFAULT '__legacy__';
            """)

            cur.execute("""
                ALTER TABLE pdf_chunks_gemini
                ADD COLUMN IF NOT EXISTS canvas_id TEXT
                NOT NULL DEFAULT 'default';
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_chunks_tenant_source
                ON pdf_chunks_gemini (
                    user_id,
                    canvas_id,
                    source
                );
            """)

            # Fast lookup by PDF filename
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_chunks_source
                ON pdf_chunks_gemini (source);
            """)

            # HNSW vector index — cosine similarity, works on empty tables
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_chunks_embedding
                ON pdf_chunks_gemini USING hnsw (embedding vector_cosine_ops);
            """)

            # ── Artwork images table ──────────────────────────────
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS artwork_images (
                    id          SERIAL       PRIMARY KEY,
                    filename    VARCHAR(500) NOT NULL UNIQUE,
                    filepath    TEXT         NOT NULL,
                    description TEXT         NOT NULL,
                    embedding   vector({EMBEDDING_DIM}),
                    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_artworks_embedding
                ON artwork_images USING hnsw (embedding vector_cosine_ops);
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
    user_id: str,
    canvas_id: str = "default",
) -> int:
    """
    Store chunks belonging to one user and one canvas.

    Re-uploading the same filename only replaces that user's
    copy inside the same canvas.
    """

    if not user_id:
        raise ValueError("user_id is required when inserting chunks.")

    safe_canvas_id = canvas_id or "default"

    conn = get_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM pdf_chunks_gemini
                WHERE user_id = %s
                  AND canvas_id = %s
                  AND source = %s;
                """,
                (
                    user_id,
                    safe_canvas_id,
                    source,
                ),
            )

            rows = [
                (
                    user_id,
                    safe_canvas_id,
                    source,
                    index,
                    chunk,
                    embeddings[index],
                )
                for index, chunk in enumerate(chunks)
            ]

            from psycopg2.extras import execute_values

            execute_values(
                cur,
                """
                INSERT INTO pdf_chunks_gemini (
                    user_id,
                    canvas_id,
                    source,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector)",
            )

        conn.commit()
        return len(rows)

    finally:
        conn.close()


def search_chunks(
    query_embedding: List[float],
    user_id: str,
    canvas_id: str = "default",
    top_k: int = 3,
    source_filters: Optional[List[str]] = None,
) -> List[dict]:
    """
    Search only inside one user's canvas.

    source_filters supports one or multiple selected files.
    """

    if not user_id:
        raise ValueError("user_id is required when searching chunks.")

    safe_canvas_id = canvas_id or "default"

    cleaned_sources = [
        str(source).strip()
        for source in (source_filters or [])
        if str(source).strip()
    ]

    conn = get_conn()

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            if cleaned_sources:
                cur.execute(
                    """
                    SELECT
                        source,
                        chunk_index,
                        content,
                        1 - (
                            embedding <=> %s::vector
                        ) AS score
                    FROM pdf_chunks_gemini
                    WHERE user_id = %s
                      AND canvas_id = %s
                      AND source = ANY(%s)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (
                        query_embedding,
                        user_id,
                        safe_canvas_id,
                        cleaned_sources,
                        query_embedding,
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        source,
                        chunk_index,
                        content,
                        1 - (
                            embedding <=> %s::vector
                        ) AS score
                    FROM pdf_chunks_gemini
                    WHERE user_id = %s
                      AND canvas_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (
                        query_embedding,
                        user_id,
                        safe_canvas_id,
                        query_embedding,
                        top_k,
                    ),
                )

            return [
                dict(row)
                for row in cur.fetchall()
            ]

    finally:
        conn.close()


def list_sources(
    user_id: str,
    canvas_id: str = "default",
) -> List[str]:
    """Return sources belonging only to one user's canvas."""

    if not user_id:
        raise ValueError("user_id is required when listing sources.")

    safe_canvas_id = canvas_id or "default"

    conn = get_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT source
                FROM pdf_chunks_gemini
                WHERE user_id = %s
                  AND canvas_id = %s
                ORDER BY source;
                """,
                (
                    user_id,
                    safe_canvas_id,
                ),
            )

            return [
                row[0]
                for row in cur.fetchall()
            ]

    finally:
        conn.close()


def get_stats(
    user_id: str,
    canvas_id: str = "default",
) -> dict:
    """Return statistics only for one user's canvas."""

    if not user_id:
        raise ValueError("user_id is required when getting stats.")

    safe_canvas_id = canvas_id or "default"

    conn = get_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM pdf_chunks_gemini
                WHERE user_id = %s
                  AND canvas_id = %s;
                """,
                (
                    user_id,
                    safe_canvas_id,
                ),
            )

            total = cur.fetchone()[0]

            cur.execute(
                """
                SELECT DISTINCT source
                FROM pdf_chunks_gemini
                WHERE user_id = %s
                  AND canvas_id = %s
                ORDER BY source;
                """,
                (
                    user_id,
                    safe_canvas_id,
                ),
            )

            sources = [
                row[0]
                for row in cur.fetchall()
            ]

            return {
                "total_chunks": total,
                "sources": sources,
            }

    finally:
        conn.close()


def delete_source(
    source_name: str,
    user_id: str,
    canvas_id: str = "default",
) -> int:
    """Delete a source only from one user's canvas."""

    if not user_id:
        raise ValueError("user_id is required when deleting a source.")

    safe_canvas_id = canvas_id or "default"

    conn = get_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM pdf_chunks_gemini
                WHERE source = %s
                  AND user_id = %s
                  AND canvas_id = %s;
                """,
                (
                    source_name,
                    user_id,
                    safe_canvas_id,
                ),
            )

            deleted = cur.rowcount

        conn.commit()
        return deleted

    finally:
        conn.close()


# ── Artwork image CRUD ────────────────────────────────────────
def insert_artwork(
    filename   : str,
    filepath   : str,
    description: str,
    embedding  : List[float],
) -> int:
    """
    Insert or replace an artwork image record.
    Returns the new row id.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Replace if same filename already exists
            cur.execute("DELETE FROM artwork_images WHERE filename = %s;", (filename,))
            cur.execute(
                """
                INSERT INTO artwork_images (filename, filepath, description, embedding)
                VALUES (%s, %s, %s, %s::vector)
                RETURNING id;
                """,
                (filename, filepath, description, embedding),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        conn.close()


def search_artworks(
    query_embedding: List[float],
    top_k: int = 3,
) -> List[dict]:
    """
    Find the most visually/semantically similar artworks
    using cosine similarity on their LLaVA descriptions.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT filename, filepath, description,
                       1 - (embedding <=> %s::vector) AS score
                FROM   artwork_images
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s;
                """,
                (query_embedding, query_embedding, top_k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def list_artworks() -> List[dict]:
    """Return all stored artwork records (without embeddings)."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, filepath, description,
                       created_at
                FROM   artwork_images
                ORDER  BY created_at DESC;
                """
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_artwork(filename: str) -> int:
    """Delete an artwork by filename. Returns rows deleted."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM artwork_images WHERE filename = %s;", (filename,)
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    cli_user_id = os.getenv(
        "NEXO_CLI_USER_ID",
        "local-cli",
    )

    cli_canvas_id = os.getenv(
        "NEXO_CLI_CANVAS_ID",
        "default",
    )

    stats = get_stats(
        user_id=cli_user_id,
        canvas_id=cli_canvas_id,
    )

    print(f"User         : {cli_user_id}")
    print(f"Canvas       : {cli_canvas_id}")
    print(f"Total chunks : {stats['total_chunks']}")
    print(f"Sources      : {stats['sources'] or 'none'}")