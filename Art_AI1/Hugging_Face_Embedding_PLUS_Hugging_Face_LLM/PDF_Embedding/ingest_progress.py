"""
In-memory progress for document ingestion (one uvicorn worker).

The /ingest request runs the pipeline in a worker thread. That thread
binds a job id with `bind(job_id)`; pipeline steps call `report(...)`
without knowing about jobs. GET /ingest/progress/{job_id} reads it.

Stages and the share of the ring they fill:
    reading   0-30 %   (pages read / OCR'd)
    embedding 30-92 %  (chunks embedded)
    saving    92-99 %
    done      100 %
"""

import re
import threading
import time
from typing import Optional

_lock = threading.Lock()
_jobs: dict = {}
_local = threading.local()

JOB_TTL_SECONDS = 15 * 60
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

STAGE_RANGES = {
    "queued": (0, 2),
    "reading": (2, 30),
    "ocr": (2, 30),
    "embedding": (30, 92),
    "saving": (92, 99),
    "done": (100, 100),
    "error": (100, 100),
}


def valid_job_id(job_id: Optional[str]) -> bool:
    return bool(job_id) and bool(_JOB_ID.match(job_id))


def _cleanup(now: float) -> None:
    for key in [key for key, job in _jobs.items() if now - job["updated"] > JOB_TTL_SECONDS]:
        del _jobs[key]


def start(job_id: str, user_id: str, filename: str) -> None:
    if not valid_job_id(job_id):
        return
    now = time.time()
    with _lock:
        _cleanup(now)
        _jobs[job_id] = {
            "user_id": user_id,
            "file": filename,
            "stage": "queued",
            "done": 0,
            "total": 0,
            "percent": 1,
            "updated": now,
        }


def bind(job_id: Optional[str]) -> None:
    """Attach the current thread to a job (None detaches)."""
    _local.job_id = job_id if valid_job_id(job_id) else None


def report(stage: str, done: int = 0, total: int = 0) -> None:
    """Called by pipeline steps; does nothing outside a bound job."""
    job_id = getattr(_local, "job_id", None)
    if job_id:
        set_stage(job_id, stage, done, total)


def set_stage(job_id: str, stage: str, done: int = 0, total: int = 0) -> None:
    low, high = STAGE_RANGES.get(stage, (0, 99))
    fraction = (done / total) if total else 0
    percent = round(low + (high - low) * max(0.0, min(1.0, fraction)))
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        # Never move the ring backwards (e.g. OCR fallback after reading).
        job.update(
            stage=stage,
            done=done,
            total=total,
            percent=max(job["percent"], percent) if stage != "error" else job["percent"],
            updated=time.time(),
        )


def get(job_id: str, user_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        if not job or job["user_id"] != user_id:
            return None
        return {key: value for key, value in job.items() if key not in ("user_id", "updated")}
