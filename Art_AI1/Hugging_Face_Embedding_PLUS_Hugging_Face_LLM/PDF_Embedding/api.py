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
    Security,
    UploadFile,
)
from fastapi.security.api_key import APIKeyHeader
from starlette.concurrency import run_in_threadpool
import uvicorn

from db import (
    delete_source as db_delete_source,
    get_stats,
    init_db,
    list_sources,
)
from embed_pipeline import ingest_pdf
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
    question: str = Form(...),
    chat_history: str = Form("[]"),
    domain: str = Form("auto"),
    task: str = Form("answer"),
):
    history = parse_chat_history(chat_history)

    try:
        result = query_general(
            question,
            chat_history=history,
            domain=domain,
            task=task,
        )
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
    question: str = Form(...),
    user_id: str = Form(...),
    canvas_id: str = Form("default"),
    top_k: int = Form(5),
    source_filters: str = Form("[]"),
    source_filter: Optional[str] = Form(None),
    chat_history: str = Form("[]"),
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

    try:
        result = query_text_rag(
            question=question,
            user_id=user_id,
            canvas_id=canvas_id,
            top_k=safe_top_k,
            source_filters=selected_sources,
            chat_history=history,
        )

    except Exception as error:
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
        sources.append(
            {
                "citation_id": f"S{index + 1}",
                "file": chunk.get("source"),
                "chunk": int(chunk.get("chunk_idx", 0)) + 1,
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

@app.post(
    "/ingest",
    summary="Upload and index a PDF document",
    dependencies=[Depends(require_api_key)],
)
async def ingest_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    canvas_id: str = Form("default"),
    ocr_mode: str = Form("printed"),
):
    safe_user_id = user_id.strip()
    safe_canvas_id = canvas_id.strip() or "default"

    if not safe_user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required.",
        )

    original_filename = Path(
        file.filename or "uploaded.pdf"
    ).name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are currently accepted for AI indexing.",
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
                return ingest_pdf(
                    pdf_path=str(tmp_path),
                    user_id=safe_user_id,
                    canvas_id=safe_canvas_id,
                    ocr_mode=ocr_mode,
                )

            print(
                f"[INGEST START] "
                f"user={safe_user_id} "
                f"canvas={safe_canvas_id} "
                f"file={original_filename}"
            )

            chunks_added = await run_in_threadpool(run_ingest)

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
        raise

    except Exception as error:
        print(
            f"[INGEST ERROR] "
            f"user={safe_user_id} "
            f"canvas={safe_canvas_id} "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}",
        )

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