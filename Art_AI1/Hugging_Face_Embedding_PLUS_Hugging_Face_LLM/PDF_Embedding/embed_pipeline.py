import os
import sys
from pathlib import Path
from typing import List

# Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

sys.path.insert(
    0,
    str(Path(__file__).parent),
)

from pdf_pipeline import process_pdf
from db import (
    init_db,
    insert_chunks,
    get_stats,
)
from gemini_embedding import (
    EMBEDDING_MODEL,
    embed_documents,
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class GeminiEmbeddingVector(list):
    """
    Compatibility wrapper.

    Older parts of the project call .tolist() because the old
    SentenceTransformer returned NumPy arrays.
    """

    def tolist(self):
        return list(self)


class GeminiEmbeddingModel:
    """
    Temporary compatibility adapter for older modules.

    This keeps image_pipeline.py from breaking while we update
    that module in a later step.
    """

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ):
        del normalize_embeddings
        del show_progress_bar
        del batch_size

        vectors = embed_documents(
            list(texts)
        )

        return [
            GeminiEmbeddingVector(vector)
            for vector in vectors
        ]


_embedding_adapter = GeminiEmbeddingModel()


def get_embed_model():
    """
    Compatibility function for existing modules.

    It no longer loads BGE, SentenceTransformer, Torch or CUDA.
    """

    return _embedding_adapter


def embed_texts(
    texts: List[str],
    title: str = None,
) -> List[List[float]]:
    """
    Create Gemini RETRIEVAL_DOCUMENT embeddings.
    """

    return embed_documents(
        texts=texts,
        title=title,
    )


def ingest_pdf(
    pdf_path: str,
    user_id: str,
    canvas_id: str = "default",
    ocr_mode: str = "printed",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Extract, chunk, embed and store one PDF.

    Re-ingesting a PDF with the same filename replaces its
    existing Gemini chunks.
    """

    if not user_id or not str(user_id).strip():
        raise ValueError(
            "user_id is required when ingesting a document."
        )

    safe_user_id = str(user_id).strip()
    safe_canvas_id = str(canvas_id or "default").strip() or "default"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    source_name = Path(pdf_path).name

    print()
    print(
        f"[Embed] Ingesting: {source_name}"
    )
    print(
        f"[Embed] OCR mode: {ocr_mode}"
    )
    print(
        f"[Embed] Model: {EMBEDDING_MODEL}"
    )
    print(
        f"[Embed] User: {safe_user_id}"
    )
    print(
        f"[Embed] Canvas: {safe_canvas_id}"
    )

    # Step 1: extract and chunk
    chunks = process_pdf(
        pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_mode=ocr_mode,
    )

    if not chunks:
        print(
            "[Embed] No text chunks were extracted."
        )
        return 0

    print(
        f"[Embed] Extracted {len(chunks)} chunks."
    )

    # Step 2: Gemini embeddings
    embeddings = embed_texts(
        texts=chunks,
        title=source_name,
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of document chunks."
        )

    # Step 3: PostgreSQL storage
    init_db()

    stored_count = insert_chunks(
        source=source_name,
        chunks=chunks,
        embeddings=embeddings,
        user_id=safe_user_id,
        canvas_id=safe_canvas_id,
    )

    stats = get_stats(
        user_id=safe_user_id,
        canvas_id=safe_canvas_id,
    )

    print()
    print(
        f"[Embed] Stored {stored_count} chunks."
    )
    print(
        f"[Embed] Gemini table total: "
        f"{stats['total_chunks']} chunks."
    )

    return stored_count


def get_db_stats(
    user_id: str,
    canvas_id: str = "default",
) -> dict:
    if not user_id or not str(user_id).strip():
        raise ValueError(
            "user_id is required when getting database stats."
        )

    return get_stats(
        user_id=str(user_id).strip(),
        canvas_id=str(canvas_id or "default").strip() or "default",
    )


if __name__ == "__main__":
    cli_user_id = os.getenv(
        "NEXO_CLI_USER_ID",
        "local-cli",
    )

    cli_canvas_id = os.getenv(
        "NEXO_CLI_CANVAS_ID",
        "default",
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "python embed_pipeline.py <file.pdf>"
        )
        print(
            "python embed_pipeline.py "
            "<file.pdf> cursive"
        )
        print(
            "python embed_pipeline.py --stats"
        )
        sys.exit(1)

    if sys.argv[1] == "--stats":
        init_db()

        stats = get_db_stats(
            user_id=cli_user_id,
            canvas_id=cli_canvas_id,
        )

        print()
        print("Gemini PostgreSQL Stats")
        print(
            f"Total chunks: "
            f"{stats['total_chunks']}"
        )
        print(
            f"Sources: "
            f"{stats['sources'] or 'none'}"
        )

        sys.exit(0)

    pdf_file = sys.argv[1]
    mode = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "printed"
    )

    count = ingest_pdf(
        pdf_path=pdf_file,
        user_id=cli_user_id,
        canvas_id=cli_canvas_id,
        ocr_mode=mode,
    )

    stats = get_db_stats(
        user_id=cli_user_id,
        canvas_id=cli_canvas_id,
    )

    print()
    print("=" * 60)
    print(
        f"Ingested: {count} chunks"
    )
    print(
        f"Source: {Path(pdf_file).name}"
    )
    print(
        f"Database total: "
        f"{stats['total_chunks']}"
    )
    print(
        f"Sources: {stats['sources']}"
    )
    print("=" * 60)