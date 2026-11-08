import math
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

from ingest_progress import report as report_progress


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

BATCH_EMBEDDING_API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{EMBEDDING_MODEL}:batchEmbedContents"
)

# The Gemini API accepts at most 100 texts per batchEmbedContents call.
# A smaller default keeps one batch of ~1000-character chunks well under
# the per-minute token limit (it was 30K TPM on our free tier).
BATCH_SIZE = max(
    1,
    min(100, int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "45"))),
)


# Free-tier Gemini counts EVERY text inside a batch as one request
# against the per-minute limit (it was 100/min for embeddings). We pace
# ourselves below that so large documents slow down instead of failing.
ITEMS_PER_MINUTE = max(
    1,
    int(os.getenv("GEMINI_EMBEDDING_ITEMS_PER_MINUTE", "90")),
)

# A batch can never be larger than what one minute allows.
BATCH_SIZE = min(BATCH_SIZE, ITEMS_PER_MINUTE)

MAX_RETRY_WAIT_SECONDS = 70


class GeminiEmbeddingError(RuntimeError):
    """Gemini Embedding request failed."""


class GeminiQuotaError(GeminiEmbeddingError):
    """
    The Gemini quota is exhausted.
    daily=True means the per-day limit: retrying today will not help.
    """

    def __init__(self, message: str, daily: bool = False):
        super().__init__(message)
        self.daily = daily


class _MinuteWindow:
    """Thread-safe sliding 60-second budget shared by ingestion and queries."""

    def __init__(self, limit: int):
        self.limit = limit
        self.events = deque()  # (timestamp, item_count)
        self.lock = threading.Lock()

    def acquire(self, count: int) -> None:
        count = min(count, self.limit)

        while True:
            with self.lock:
                now = time.monotonic()

                while self.events and now - self.events[0][0] >= 60:
                    self.events.popleft()

                used = sum(items for _, items in self.events)

                if used + count <= self.limit:
                    self.events.append((now, count))
                    return

                wait = 60 - (now - self.events[0][0]) + 0.5

            print(f"[Embedding] Pacing for the per-minute quota: waiting {wait:.0f}s...")
            time.sleep(max(0.5, wait))


_minute_window = _MinuteWindow(ITEMS_PER_MINUTE)


def _retry_delay_seconds(response: "requests.Response") -> Optional[float]:
    """Read Retry-After or Google's RetryInfo.retryDelay (e.g. "41s")."""

    header = response.headers.get("Retry-After")

    try:
        if header is not None:
            return float(header)
    except ValueError:
        pass

    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return None

    for detail in details:
        delay = str(detail.get("retryDelay", ""))
        match = re.match(r"^([\d.]+)s$", delay)
        if match:
            return float(match.group(1))

    return None


def _is_daily_quota(message: str) -> bool:
    lowered = message.lower()
    return "perday" in lowered or "per day" in lowered or "requests_per_day" in lowered


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

    del max_retries  # retries and quota pacing live in _post_with_retries

    data = _post_with_retries(EMBEDDING_API_URL, payload, item_count=1)

    vector = data.get("embedding", {}).get("values", [])

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


def _post_with_retries(
    url: str,
    payload: dict,
    item_count: int = 1,
    max_retries: int = 6,
) -> dict:
    """
    POST to Gemini, staying under the per-minute quota.

    - Waits for room in the shared 60 s budget before every call.
    - On 429 waits as long as Google asks (retryDelay), up to 70 s.
    - A per-DAY quota error is raised at once (retrying cannot help).
    """

    for attempt in range(max_retries):
        _minute_window.acquire(item_count)

        try:
            response = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json=payload,
                timeout=120,
            )
        except requests.RequestException as error:
            if attempt == max_retries - 1:
                raise GeminiEmbeddingError(
                    f"Could not reach Gemini Embedding: {error}"
                ) from error

            time.sleep(2 ** attempt)
            continue

        if response.ok:
            return response.json()

        message = _read_error_message(response)

        if response.status_code == 429:
            if _is_daily_quota(message):
                raise GeminiQuotaError(message, daily=True)

            if attempt == max_retries - 1:
                raise GeminiQuotaError(message, daily=False)

            wait_seconds = min(
                MAX_RETRY_WAIT_SECONDS,
                (_retry_delay_seconds(response) or 20 * (attempt + 1)) + 1,
            )

            print(
                f"[Embedding] Quota hit (429). Waiting {wait_seconds:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_seconds)
            continue

        if response.status_code >= 500 and attempt < max_retries - 1:
            time.sleep(min(30, 2 ** (attempt + 1)))
            continue

        raise GeminiEmbeddingError(
            f"Gemini Embedding returned HTTP "
            f"{response.status_code}: {message}"
        )

    raise GeminiEmbeddingError("Gemini Embedding request failed.")


def _embed_batch(
    texts: List[str],
    title: Optional[str],
) -> List[List[float]]:
    requests_payload = []

    for text in texts:
        item = {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": EMBEDDING_DIM,
        }

        if title:
            item["title"] = title.strip()

        requests_payload.append(item)

    data = _post_with_retries(
        BATCH_EMBEDDING_API_URL,
        {"requests": requests_payload},
        item_count=len(texts),
    )

    embeddings = data.get("embeddings") or []

    if len(embeddings) != len(texts):
        raise GeminiEmbeddingError(
            "Gemini batch returned "
            f"{len(embeddings)} embeddings for {len(texts)} texts."
        )

    vectors = []

    for embedding in embeddings:
        vector = embedding.get("values") or []

        if len(vector) != EMBEDDING_DIM:
            raise GeminiEmbeddingError(
                "Unexpected embedding dimension: "
                f"expected {EMBEDDING_DIM}, received {len(vector)}."
            )

        vectors.append(normalize_vector(vector))

    return vectors


def embed_documents(
    texts: List[str],
    title: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed document chunks with batchEmbedContents.

    Before: one HTTP request per chunk plus a 4.1 s sleep, so a 57-chunk
    PDF needed 57 requests and about 4 minutes, which caused upload
    timeouts and used up the daily request quota.
    Now: ceil(N / BATCH_SIZE) requests (57 chunks -> 2 requests).

    Output order always matches input order.
    """

    if not GEMINI_API_KEY:
        raise GeminiEmbeddingError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    cleaned = [str(text).strip() for text in texts]

    if any(not text for text in cleaned):
        raise ValueError(
            "Cannot create an embedding from empty text."
        )

    total = len(cleaned)
    embeddings: List[List[float]] = []

    # Optional pause between batches (default 0). Raise it if the
    # per-minute token quota is hit on very large documents.
    delay_seconds = float(
        os.getenv("GEMINI_EMBEDDING_BATCH_DELAY_SECONDS", "0")
    )

    report_progress("embedding", 0, total)

    for start in range(0, total, BATCH_SIZE):
        batch = cleaned[start:start + BATCH_SIZE]
        end = start + len(batch)

        print(
            f"[Embedding] Batch {start + 1}-{end} of {total}"
        )

        embeddings.extend(_embed_batch(batch, title))
        report_progress("embedding", end, total)

        if end < total and delay_seconds > 0:
            time.sleep(delay_seconds)

    return embeddings
