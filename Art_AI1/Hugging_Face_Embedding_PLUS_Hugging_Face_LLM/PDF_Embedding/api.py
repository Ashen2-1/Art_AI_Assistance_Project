############################api.py
import hmac
import json
import os
import tempfile

from prompt_profiles import normalize_domain
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.security.api_key import APIKeyHeader
from starlette.concurrency import run_in_threadpool
import uvicorn

from db import (
    delete_source as db_delete_source,
    get_source_chunks,
    get_stats,
    init_db,
    list_sources,
)
from document_pipeline import (
    SUPPORTED_EXTENSIONS,
    UnsupportedDocumentError,
    parse_location,
)
# Imported under another name: the /ingest endpoint below is itself
# called ingest_document, and would otherwise shadow this function.
from embed_pipeline import ingest_document as index_document_file
from gemini_embedding import GeminiQuotaError
import ingest_progress
import detective
from image_tools import IMAGE_EXTENSIONS, crop_region, prepare_image
import user_keys
from gemini_client import UserKeyError
from rag_query import query_general, query_text_rag


load_dotenv()

API_KEY = os.getenv("API_KEY", "").strip()
API_KEY_HEADER_NAME = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
)


def require_api_key(
    provided_key: Optional[str] = Security(api_key_header),
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="The FastAPI API_KEY is not configured.",
        )

    if (
        not provided_key
        or not hmac.compare_digest(provided_key, API_KEY)
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key.",
        )

    return provided_key


def normalize_identity(
    user_id: str,
    canvas_id: str = "default",
):
    safe_user_id = str(user_id or "").strip()
    safe_canvas_id = (
        str(canvas_id or "default").strip()
        or "default"
    )

    if not safe_user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    return safe_user_id, safe_canvas_id


def quota_error_message(error: "GeminiQuotaError") -> str:
    if error.daily:
        return (
            "The AI service has used today's free Gemini quota. The file is saved, "
            "but it can only be indexed after the quota resets (or after billing is "
            "enabled on the Gemini API key)."
        )

    return (
        "The AI service is busy (Gemini per-minute limit). Please wait about a "
        "minute and try again."
    )


def parse_chat_history(chat_history: str):
    try:
        history = json.loads(chat_history)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="chat_history must be valid JSON.",
        )

    if not isinstance(history, list):
        raise HTTPException(
            status_code=400,
            detail="chat_history must be a JSON array.",
        )

    cleaned_history = []

    for message in history[-12:]:
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role", "user")
        ).strip()

        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            continue

        cleaned_history.append(
            {
                "role": role,
                "content": content[:6000],
            }
        )

    return cleaned_history

def parse_source_filters(
    source_filters: str,
    source_filter: Optional[str] = None,
) -> List[str]:
    try:
        parsed_filters = json.loads(
            source_filters or "[]"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="source_filters must be a valid JSON array.",
        )

    if not isinstance(parsed_filters, list):
        raise HTTPException(
            status_code=400,
            detail="source_filters must be a JSON array.",
        )

    candidates = list(parsed_filters)

    if source_filter:
        candidates.append(source_filter)

    cleaned_sources = []
    seen_sources = set()

    for source in candidates:
        normalized_source = str(source).strip()

        if (
            normalized_source
            and normalized_source not in seen_sources
        ):
            seen_sources.add(normalized_source)
            cleaned_sources.append(normalized_source)

    return cleaned_sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError(
            "API_KEY is missing. Add API_KEY to the Python environment."
        )

    init_db()
    print("[NEXO RAG API] Database ready.")

    yield


app = FastAPI(
    title="NEXO Multidisciplinary RAG API",
    description=(
        "Document ingestion and grounded AI research assistance "
        "for multidisciplinary research fields."
    ),
    version="1.2.0",
    lifespan=lifespan,
)
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[REQUEST START] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        print(
            f"[REQUEST DONE] {request.method} {request.url.path} "
            f"status={response.status_code}"
        )
        return response

    except Exception as error:
        print(
            f"[REQUEST CRASH] {request.method} {request.url.path} "
            f"{type(error).__name__}: {error}"
        )
        raise

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "nexo-rag-api",
    }


@app.post(
    "/query/general",
    dependencies=[Depends(require_api_key)],

)
async def query_general_endpoint(
    request: Request,
    question: str = Form(...),
    chat_history: str = Form("[]"),
    domain: str = Form("auto"),
    task: str = Form("answer"),
):
    history = parse_chat_history(chat_history)

    try:
        with user_keys.using(user_keys.from_headers(request.headers)):
            result = query_general(
                question,
                chat_history=history,
                domain=domain,
                task=task,
            )
    except UserKeyError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        print(
            f"[GENERAL QUERY ERROR] "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="The AI service could not answer the question.",
        )

    return {
        "mode": "general_chat",
        "question": question,
        "answer": result["answer"],
        "sources": [],
    }


@app.post(
    "/query/text",
    dependencies=[Depends(require_api_key)],
)
async def query_text_endpoint(
    request: Request,
    question: str = Form(...),
    user_id: str = Form(...),
    canvas_id: str = Form("default"),
    top_k: int = Form(5),
    source_filters: str = Form("[]"),
    source_filter: Optional[str] = Form(None),
    chat_history: str = Form("[]"),
    inline_context: str = Form(""),
):
    safe_question = question.strip()
    safe_user_id = user_id.strip()
    safe_canvas_id = canvas_id.strip() or "default"
    safe_top_k = max(1, min(top_k, 10))

    if not safe_question:
        raise HTTPException(
            status_code=400,
            detail="Question is required.",
        )

    if not safe_user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    history = parse_chat_history(chat_history)

    selected_sources = parse_source_filters(
        source_filters=source_filters,
        source_filter=source_filter,
    )

    request_keys = user_keys.from_headers(request.headers)

    try:
        with user_keys.using(request_keys):
            result = query_text_rag(
                question=safe_question,
                user_id=safe_user_id,
                canvas_id=safe_canvas_id,
                top_k=safe_top_k,
                source_filters=selected_sources,
                chat_history=history,
                inline_context=inline_context,
            )

    except UserKeyError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except GeminiQuotaError as error:
        raise HTTPException(status_code=429, detail=quota_error_message(error))

    except Exception as error:
        if request_keys.get("gemini_key") and _looks_like_bad_key(error):
            raise HTTPException(status_code=400, detail=OWN_GEMINI_KEY_REJECTED)

        print(
            "[TEXT RAG ERROR] "
            f"user={safe_user_id} "
            f"canvas={safe_canvas_id} "
            f"type={type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"The RAG service failed: {error}",
        )

    sources = []

    for index, chunk in enumerate(result.get("chunks", [])):
        chunk_text = str(chunk.get("text", ""))

        sources.append(
            {
                "citation_id": f"S{index + 1}",
                "file": chunk.get("source"),
                "chunk": int(chunk.get("chunk_idx", 0)) + 1,
                "chunk_index": int(chunk.get("chunk_idx", 0)),
                # Where the passage is: page / slide / section / sheet rows / lines.
                "location": parse_location(chunk_text),
                "similarity": round(
                    float(chunk.get("score", 0)),
                    4,
                ),
                "preview": str(
                    chunk.get("text", "")
                )[:800].replace("\n", " "),
                "text": str(chunk.get("text", "")),
            }
        )

    return {
        "mode": "text_rag",
        "question": safe_question,
        "answer": result.get(
            "answer",
            "No answer was returned.",
        ),
        "sources": sources,
    }

OWN_GEMINI_KEY_REJECTED = (
    "Your Gemini API key was rejected by Google. Check it in Settings → AI keys, "
    "or remove it to use the NEXO default."
)


def _looks_like_bad_key(error: Exception) -> bool:
    text = str(error)
    return "HTTP 401" in text or "HTTP 403" in text or "API key not valid" in text


@app.post(
    "/ingest",
    summary="Upload and index a document (PDF, Office, CSV, text, code)",
    dependencies=[Depends(require_api_key)],
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    canvas_id: str = Form("default"),
    ocr_mode: str = Form("printed"),
    job_id: str = Form(""),
):
    safe_user_id = user_id.strip()
    safe_canvas_id = canvas_id.strip() or "default"
    safe_job_id = job_id.strip() if ingest_progress.valid_job_id(job_id.strip()) else ""
    request_keys = user_keys.from_headers(request.headers)

    if not safe_user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    original_filename = Path(
        file.filename or "uploaded.pdf"
    ).name

    if Path(original_filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "This file type cannot be read by the AI yet. Supported: PDF, Word, "
                "PowerPoint, Excel, CSV, text, Markdown, code, Jupyter notebooks and images."
            ),
        )

    if ocr_mode not in ("printed", "cursive"):
        raise HTTPException(
            status_code=400,
            detail="ocr_mode must be 'printed' or 'cursive'.",
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="nexo_ingest_"
        ) as temp_dir:
            tmp_path = Path(temp_dir) / original_filename

            file_content = await file.read()
            tmp_path.write_bytes(file_content)

            def run_ingest():
                # Pipeline steps report progress to this job (see ingest_progress.py).
                ingest_progress.bind(safe_job_id or None)
                try:
                    # The uploader's own Gemini key (if any) pays for the embeddings.
                    with user_keys.using(request_keys):
                        return index_document_file(
                            pdf_path=str(tmp_path),
                            user_id=safe_user_id,
                            canvas_id=safe_canvas_id,
                            ocr_mode=ocr_mode,
                        )
                finally:
                    ingest_progress.bind(None)

            if safe_job_id:
                ingest_progress.start(safe_job_id, safe_user_id, original_filename)

            print(
                f"[INGEST START] "
                f"user={safe_user_id} "
                f"canvas={safe_canvas_id} "
                f"file={original_filename}"
            )

            chunks_added = await run_in_threadpool(run_ingest)

            if safe_job_id:
                ingest_progress.set_stage(safe_job_id, "done", 1, 1)

            print(
                f"[INGEST DONE] "
                f"user={safe_user_id} "
                f"canvas={safe_canvas_id} "
                f"file={original_filename} "
                f"chunks={chunks_added}"
            )

        stats = get_stats(
            user_id=safe_user_id,
            canvas_id=safe_canvas_id,
        )

        return {
            "status": "ok",
            "file": original_filename,
            "user_id": safe_user_id,
            "canvas_id": safe_canvas_id,
            "chunks_added": chunks_added,
            "db_total": stats["total_chunks"],
            "sources": stats["sources"],
        }

    except HTTPException:
        if safe_job_id:
            ingest_progress.set_stage(safe_job_id, "error")
        raise

    except GeminiQuotaError as error:
        if safe_job_id:
            ingest_progress.set_stage(safe_job_id, "error")
        print(f"[INGEST QUOTA] {original_filename}: daily={error.daily} {error}")

        raise HTTPException(status_code=429, detail=quota_error_message(error))

    except (UnsupportedDocumentError, ValueError) as error:
        # Clear, user-facing problems (unreadable file, legacy format, ...).
        if safe_job_id:
            ingest_progress.set_stage(safe_job_id, "error")
        print(f"[INGEST REJECTED] {original_filename}: {error}")

        raise HTTPException(status_code=422, detail=str(error))

    except Exception as error:
        print(
            f"[INGEST ERROR] "
            f"user={safe_user_id} "
            f"canvas={safe_canvas_id} "
            f"{type(error).__name__}: {error}"
        )

        if safe_job_id:
            ingest_progress.set_stage(safe_job_id, "error")

        if request_keys.get("gemini_key") and _looks_like_bad_key(error):
            raise HTTPException(status_code=400, detail=OWN_GEMINI_KEY_REJECTED)

        print("========== INGEST FAILED ==========")
        print(f"file: {file.filename if file else 'unknown'}")
        print(f"user_id: {user_id}")
        print(f"canvas_id: {canvas_id}")
        print(str(error))

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}",
        )

@app.post(
    "/detective/analyze",
    dependencies=[Depends(require_api_key)],
)
async def detective_analyze(
    request: Request,
    images: List[UploadFile] = File(...),
    user_id: str = Form(...),
    canvas_id: str = Form("default"),
    question: str = Form(""),
    source_filters: str = Form("[]"),
    focus: str = Form("[]"),
):
    """
    Photo Detective against the workspace sources:
    1 image = identify, 2 = compare, 3-12 = group (who appears where).
    focus: JSON list aligned with images, each null or
           {"box": {"x","y","w","h"} as 0-1 fractions, "note": "the man on the left"}.
    """

    if not 1 <= len(images) <= detective.MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Send between 1 and {detective.MAX_IMAGES} images.")

    try:
        focus_list = json.loads(focus or "[]")
        if not isinstance(focus_list, list):
            focus_list = []
    except ValueError:
        focus_list = []

    safe_user_id = user_id.strip()
    if not safe_user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    request_keys = user_keys.from_headers(request.headers)
    selected = parse_source_filters(source_filters=source_filters, source_filter=None)

    try:
        with tempfile.TemporaryDirectory(prefix="nexo_detective_") as temp_dir:
            prepared = []
            # Many images in one request: keep each smaller so the request stays fast.
            max_side = 2048 if len(images) <= 2 else 1280 if len(images) <= 6 else 1024
            for index, upload in enumerate(images):
                name = Path(upload.filename or f"image-{index + 1}.jpg").name
                if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"{name} is not a supported image.")
                raw = Path(temp_dir) / f"raw-{index}{Path(name).suffix.lower()}"
                raw.write_bytes(await upload.read())
                item = {"path": prepare_image(str(raw), temp_dir, max_side=max_side), "name": name}
                spec = focus_list[index] if index < len(focus_list) and isinstance(focus_list[index], dict) else {}
                if isinstance(spec.get("box"), dict):
                    item["focus_path"] = crop_region(str(raw), spec["box"], temp_dir)
                if spec.get("note"):
                    item["focus_note"] = str(spec["note"])[:200]
                prepared.append(item)

            def run():
                with user_keys.using(request_keys):
                    return detective.analyze(
                        images=prepared,
                        user_id=safe_user_id,
                        canvas_id=canvas_id.strip() or "default",
                        question=question[:1000],
                        source_filters=selected,
                    )

            result = await run_in_threadpool(run)

    except HTTPException:
        raise
    except UserKeyError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except GeminiQuotaError as error:
        raise HTTPException(status_code=429, detail=quota_error_message(error))
    except Exception as error:
        if request_keys.get("gemini_key") and _looks_like_bad_key(error):
            raise HTTPException(status_code=400, detail=OWN_GEMINI_KEY_REJECTED)
        print(f"[DETECTIVE ERROR] user={safe_user_id} {type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail=f"The Photo Detective could not finish: {error}")

    sources = [
        {
            "citation_id": f"S{row['citation_id']}",
            "file": row.get("source"),
            "chunk": int(row.get("chunk_idx", 0)) + 1,
            "chunk_index": int(row.get("chunk_idx", 0)),
            "location": parse_location(str(row.get("text", ""))),
            "similarity": round(float(row.get("score", 0)), 4),
            "preview": str(row.get("text", ""))[:800].replace("\n", " "),
            "text": str(row.get("text", "")),
        }
        for row in result["evidence"]
    ]

    return {
        "mode": result["mode"],
        "labels": result.get("labels", []),
        "observations": result["observations"],
        "report": result["report"],
        "sources": sources,
    }


@app.get(
    "/ingest/progress/{job_id}",
    dependencies=[Depends(require_api_key)],
)
async def ingest_progress_endpoint(job_id: str, user_id: str):
    """Progress of an /ingest request started with this job_id."""

    job = ingest_progress.get(job_id, user_id.strip())

    if not job:
        # Not started yet (still uploading), finished long ago, or not yours.
        return {"stage": "unknown", "percent": 0}

    return job


@app.get(
    "/source/chunks",
    dependencies=[Depends(require_api_key)],
)
async def source_chunks(
    user_id: str,
    source: str,
    canvas_id: str = "default",
):
    """
    All chunks of one source, in order, for the source viewer.
    Express passes the authenticated user_id; this service is only
    reachable with the internal API key.
    """

    safe_user_id = user_id.strip()

    if not safe_user_id or not source.strip():
        raise HTTPException(status_code=400, detail="user_id and source are required.")

    rows = get_source_chunks(
        user_id=safe_user_id,
        canvas_id=canvas_id.strip() or "default",
        source=source,
    )

    return {
        "source": source,
        "chunks": [
            {
                "chunk_index": row["chunk_index"],
                "location": parse_location(row["content"]),
                "text": row["content"],
            }
            for row in rows
        ],
    }


@app.get(
    "/sources",
    dependencies=[Depends(require_api_key)],
)
async def get_sources(
    user_id: str,
    canvas_id: str = "default",
):
    return {
        "sources": list_sources(
            user_id=user_id,
            canvas_id=canvas_id,
        ),
    }


@app.get(
    "/stats",
    dependencies=[Depends(require_api_key)],
)
async def database_stats(
    user_id: str,
    canvas_id: str = "default",
):
    stats = get_stats(
        user_id=user_id,
        canvas_id=canvas_id,
    )

    return {
        "total_chunks": stats["total_chunks"],
        "sources": stats["sources"],
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 384,
    }


@app.delete(
    "/source/{source_name}",
    dependencies=[Depends(require_api_key)],
)
async def delete_source(
    source_name: str,
    user_id: str,
    canvas_id: str = "default",
):
    deleted = db_delete_source(
        source_name=source_name,
        user_id=user_id,
        canvas_id=canvas_id,
    )

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Source '{source_name}' was not found.",
        )

    return {
        "status": "deleted",
        "source": source_name,
        "chunks_removed": deleted,
    }

if __name__ == "__main__":
    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )