import json
import os
import shutil
import tempfile
import hmac

from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
)
from fastapi.security.api_key import APIKeyHeader
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

    return history


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError(
            "API_KEY is missing. Add API_KEY to the Python .env file."
        )

    init_db()
    print("[NEXO RAG API] Database ready.")
    yield


app = FastAPI(
    title="NEXO Multidisciplinary RAG API",
    description=(
        "Document ingestion and grounded AI research assistance "
        "for arts, humanities, engineering, mathematics, science, "
        "and other research fields."
    ),
    version="1.1.0",
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
):
    history = parse_chat_history(chat_history)

    try:
        result = query_general(
            question,
            chat_history=history,
        )
    except Exception as error:
        print(f"[GENERAL QUERY ERROR] {error}")
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
    top_k: int = Form(3),
    source_filter: Optional[str] = Form(None),
    chat_history: str = Form("[]"),
):
    history = parse_chat_history(chat_history)
    safe_top_k = max(1, min(top_k, 10))

    try:
        result = query_text_rag(
            question,
            top_k=safe_top_k,
            source_filter=source_filter,
            chat_history=history,
        )
    except Exception as error:
        print(f"[TEXT RAG ERROR] {error}")
        raise HTTPException(
            status_code=500,
            detail="The RAG service could not answer the question.",
        )

    return {
        "mode": "text_rag",
        "question": question,
        "answer": result["answer"],
        "citations": [
            {
                "id": index + 1,
                "file": chunk["source"],
                "chunk": chunk["chunk_idx"] + 1,
                "page": chunk.get("page") or chunk.get("page_number"),
                "similarity": round(float(chunk["score"]), 4),
                "preview": chunk["text"][:300].replace("\n", " "),
                "text": chunk["text"],
            }
            for index, chunk in enumerate(result.get("chunks", []))
        ],

        # 暂时保留 sources，避免现有 Node 和 React 前端坏掉
        "sources": [
            {
                "id": index + 1,
                "file": chunk["source"],
                "chunk": chunk["chunk_idx"] + 1,
                "page": chunk.get("page") or chunk.get("page_number"),
                "similarity": round(float(chunk["score"]), 4),
                "preview": chunk["text"][:300].replace("\n", " "),
                "text": chunk["text"],
            }
            for index, chunk in enumerate(result.get("chunks", []))
        ],
    }


@app.post("/ingest",summary="Upload a PDF and ingest it into the database",dependencies=[Depends(require_api_key)],)
async def ingest_document(
    file: UploadFile = File(...),
    ocr_mode: str = Form("printed"),
):
    original_filename = Path(file.filename or "uploaded.pdf").name

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    if ocr_mode not in ("printed", "cursive"):
        raise HTTPException(
            status_code=400,
            detail="ocr_mode must be 'printed' or 'cursive'.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="nexo_ingest_") as temp_dir:
            # 临时目录是随机的，但文件本身保留用户上传的原文件名
            tmp_path = Path(temp_dir) / original_filename

            file_content = await file.read()
            tmp_path.write_bytes(file_content)

            # 放进工作线程，避免阻塞 FastAPI 的 /health
            chunks_added = await run_in_threadpool(
                ingest_pdf,
                str(tmp_path),
                ocr_mode=ocr_mode,
            )

        stats = get_stats()

        return {
            "status": "ok",
            "file": original_filename,
            "chunks_added": chunks_added,
            "db_total": stats["total_chunks"],
            "sources": stats["sources"],
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"[INGEST ERROR] {type(error).__name__}: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}",
        )

@app.get(
    "/sources",
    dependencies=[Depends(require_api_key)],
)
async def get_sources():
    return {
        "sources": list_sources(),
    }


@app.get(
    "/stats",
    dependencies=[Depends(require_api_key)],
)
async def database_stats():
    stats = get_stats()

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
async def delete_source(source_name: str):
    deleted = db_delete_source(source_name)

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
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )