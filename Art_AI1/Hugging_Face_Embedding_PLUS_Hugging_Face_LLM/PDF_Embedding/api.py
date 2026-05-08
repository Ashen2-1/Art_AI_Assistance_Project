################################################################
### Art AI — FastAPI Server
###
### Endpoints:
###   POST   /ingest              Upload PDF -> embed -> store
###   POST   /query/text          Question -> RAG answer
###   POST   /query/vlm           Image + question -> LLaVA answer
###   POST   /query/hybrid        Image + question + chunks -> answer
###   GET    /sources             List all ingested PDFs
###   GET    /stats               DB statistics
###   DELETE /source/{name}       Remove a PDF from the DB
###
### Run:
###   python api.py
###   (or)  uvicorn api:app --reload --port 8000
################################################################

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# local modules
sys.path.insert(0, str(Path(__file__).parent))
from db import init_db, get_stats, list_sources, delete_source as db_delete
from embed_pipeline import ingest_pdf
from rag_query import query_text_rag, query_vlm_only, query_hybrid

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title       = "Art AI RAG API",
    description = "PDF ingestion + LLaVA-powered art history research assistant",
    version     = "1.0.0",
)


@app.on_event("startup")
def startup():
    """Ensure DB tables exist when the server starts."""
    init_db()
    print("[API] Server ready.")


# ── POST /ingest ──────────────────────────────────────────────
@app.post("/ingest", summary="Upload a PDF and ingest it into the database")
async def ingest(
    file    : UploadFile = File(...,  description="PDF file to ingest"),
    ocr_mode: str        = Form("printed", description="printed | cursive"),
):
    """
    Upload a PDF file.
    - Extracts and chunks the text (pdfplumber or OCR)
    - Embeds chunks with BGE-small
    - Stores in PostgreSQL + pgvector

    Re-uploading the same filename replaces the old data.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if ocr_mode not in ("printed", "cursive"):
        raise HTTPException(status_code=400, detail="ocr_mode must be 'printed' or 'cursive'.")

    # Save upload to a temp file
    tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        n = ingest_pdf(tmp_path, ocr_mode=ocr_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    stats = get_stats()
    return {
        "status"      : "ok",
        "file"        : file.filename,
        "chunks_added": n,
        "db_total"    : stats["total_chunks"],
        "sources"     : stats["sources"],
    }


# ── POST /query/text ──────────────────────────────────────────
@app.post("/query/text", summary="Ask a question answered from ingested PDFs")
async def query_text(
    question     : str           = Form(..., description="Your research question"),
    top_k        : int           = Form(3,   description="Number of chunks to retrieve"),
    source_filter: Optional[str] = Form(None, description="Limit to one PDF filename"),
):
    """
    TEXT RAG MODE:
    Retrieves the most relevant PDF chunks and asks LLaVA to answer
    strictly from those chunks. No image required.
    """
    try:
        result = query_text_rag(question, top_k=top_k, source_filter=source_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "mode"    : "text_rag",
        "question": question,
        "answer"  : result["answer"],
        "sources" : [
            {
                "file"      : c["source"],
                "chunk"     : c["chunk_idx"] + 1,
                "similarity": c["score"],
                "preview"   : c["text"][:200].replace("\n", " "),
            }
            for c in result.get("chunks", [])
        ],
    }


# ── POST /query/vlm ───────────────────────────────────────────
@app.post("/query/vlm", summary="Analyze an artwork image with LLaVA")
async def query_vlm(
    question: str        = Form(..., description="Your question about the image"),
    image   : UploadFile = File(..., description="Artwork image (jpg, png, etc.)"),
):
    """
    VLM-ONLY MODE:
    Passes the image directly to LLaVA. No text retrieval.
    Best for visual analysis, style description, composition.
    """
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"vlm_{image.filename}"
    )
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = query_vlm_only(question, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {
        "mode"    : "vlm_only",
        "question": question,
        "answer"  : result["answer"],
    }


# ── POST /query/hybrid ────────────────────────────────────────
@app.post("/query/hybrid", summary="Analyze image + retrieved text chunks together")
async def query_hybrid_endpoint(
    question     : str           = Form(...,  description="Your research question"),
    image        : UploadFile    = File(...,  description="Artwork image"),
    top_k        : int           = Form(3,    description="Number of chunks to retrieve"),
    source_filter: Optional[str] = Form(None, description="Limit to one PDF filename"),
):
    """
    HYBRID MODE:
    LLaVA sees both the artwork image AND the relevant PDF chunks.
    Most powerful mode for connecting visual art to research literature.
    """
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"hybrid_{image.filename}"
    )
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = query_hybrid(
            question, tmp_path,
            top_k=top_k,
            source_filter=source_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {
        "mode"    : "hybrid",
        "question": question,
        "answer"  : result["answer"],
        "sources" : [
            {
                "file"      : c["source"],
                "chunk"     : c["chunk_idx"] + 1,
                "similarity": c["score"],
                "preview"   : c["text"][:200].replace("\n", " "),
            }
            for c in result.get("chunks", [])
        ],
    }


# ── GET /sources ──────────────────────────────────────────────
@app.get("/sources", summary="List all ingested PDF filenames")
async def sources():
    return {"sources": list_sources()}


# ── GET /stats ────────────────────────────────────────────────
@app.get("/stats", summary="Database statistics")
async def stats():
    s = get_stats()
    return {
        "total_chunks": s["total_chunks"],
        "sources"     : s["sources"],
        "database"    : "PostgreSQL 16 + pgvector 0.8.2",
    }


# ── DELETE /source/{name} ─────────────────────────────────────
@app.delete("/source/{source_name}", summary="Remove a PDF from the database")
async def delete_source_endpoint(source_name: str):
    """
    Deletes all chunks for the given PDF filename.
    Example: DELETE /source/Sigmund.pdf
    """
    deleted = db_delete(source_name)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"'{source_name}' not found in database.",
        )
    return {
        "status" : "deleted",
        "source" : source_name,
        "removed": deleted,
    }


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
