import math
import os
import time
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
).strip()

EMBEDDING_DIM = int(
    os.getenv(
        "GEMINI_EMBEDDING_DIM",
        "384",
    )
)

EMBEDDING_API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{EMBEDDING_MODEL}:embedContent"
)


class GeminiEmbeddingError(RuntimeError):
    """Gemini Embedding request failed."""


def normalize_vector(
    vector: List[float],
) -> List[float]:
    """
    Normalize the vector to length 1.

    gemini-embedding-001 requires manual normalization when
    using a reduced dimension such as 384.
    """

    magnitude = math.sqrt(
        sum(value * value for value in vector)
    )

    if magnitude == 0:
        raise GeminiEmbeddingError(
            "Gemini returned a zero-length embedding."
        )

    return [
        value / magnitude
        for value in vector
    ]


def _read_error_message(
    response: requests.Response,
) -> str:
    try:
        data = response.json()

        return (
            data.get("error", {}).get("message")
            or data.get("message")
            or response.text
        )
    except ValueError:
        return (
            response.text
            or f"HTTP {response.status_code}"
        )


def embed_text(
    text: str,
    task_type: str,
    title: Optional[str] = None,
    max_retries: int = 4,
) -> List[float]:
    """
    Generate one normalized Gemini text embedding.

    task_type should normally be:

    RETRIEVAL_DOCUMENT:
        For PDF chunks and stored source materials.

    RETRIEVAL_QUERY:
        For user search questions.
    """

    if not GEMINI_API_KEY:
        raise GeminiEmbeddingError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    cleaned_text = str(text).strip()

    if not cleaned_text:
        raise ValueError(
            "Cannot create an embedding from empty text."
        )

    valid_task_types = {
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
    }

    if task_type not in valid_task_types:
        raise ValueError(
            f"Unsupported embedding task type: {task_type}"
        )

    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [
                {
                    "text": cleaned_text,
                }
            ]
        },
        "taskType": task_type,
        "outputDimensionality": EMBEDDING_DIM,
    }

    if (
        title
        and task_type == "RETRIEVAL_DOCUMENT"
    ):
        payload["title"] = title.strip()

    for attempt in range(max_retries):
        try:
            response = requests.post(
                EMBEDDING_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json=payload,
                timeout=60,
            )
        except requests.Timeout as error:
            if attempt == max_retries - 1:
                raise GeminiEmbeddingError(
                    "Gemini Embedding request timed out."
                ) from error

            time.sleep(2 ** attempt)
            continue

        except requests.RequestException as error:
            if attempt == max_retries - 1:
                raise GeminiEmbeddingError(
                    "Could not connect to Gemini Embedding."
                ) from error

            time.sleep(2 ** attempt)
            continue

        if response.ok:
            data = response.json()

            vector = (
                data.get("embedding", {})
                .get("values", [])
            )

            if not vector:
                raise GeminiEmbeddingError(
                    "Gemini returned no embedding values."
                )

            if len(vector) != EMBEDDING_DIM:
                raise GeminiEmbeddingError(
                    "Unexpected embedding dimension: "
                    f"expected {EMBEDDING_DIM}, "
                    f"received {len(vector)}."
                )

            return normalize_vector(vector)

        # Retry temporary quota/server errors
        if (
            response.status_code == 429
            or response.status_code >= 500
        ):
            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt

                print(
                    "[Embedding] Temporary API error. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

        message = _read_error_message(response)

        raise GeminiEmbeddingError(
            f"Gemini Embedding returned HTTP "
            f"{response.status_code}: {message}"
        )

    raise GeminiEmbeddingError(
        "Gemini Embedding request failed."
    )


def embed_document(
    text: str,
    title: Optional[str] = None,
) -> List[float]:
    """
    Embed a document chunk that will be stored in PostgreSQL.
    """

    return embed_text(
        text=text,
        task_type="RETRIEVAL_DOCUMENT",
        title=title,
    )


def embed_query(
    question: str,
) -> List[float]:
    """
    Embed a user question used to search stored documents.
    """

    return embed_text(
        text=question,
        task_type="RETRIEVAL_QUERY",
    )


def embed_documents(
    texts: List[str],
    title: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed several document chunks sequentially.

    A small pause helps avoid free-tier rate limits.
    """

    embeddings = []
    total = len(texts)

    for index, text in enumerate(
        texts,
        start=1,
    ):
        print(
            f"[Embedding] Processing "
            f"{index}/{total}"
        )

        vector = embed_document(
            text=text,
            title=title,
        )

        embeddings.append(vector)

        if index < total:
            delay_seconds = float(
                os.getenv(
                    "GEMINI_EMBEDDING_DELAY_SECONDS",
                    "4.1",
                )
            )

            time.sleep(delay_seconds)

    return embeddings