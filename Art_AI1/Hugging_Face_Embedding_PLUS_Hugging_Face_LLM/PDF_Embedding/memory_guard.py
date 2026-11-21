"""
Memory guard for the 512 MB Render instance.

- heavy_slot(): only HEAVY_JOBS (default 1) document reads / OCR runs at a
  time. Two big PDFs read at once used to push the server over its memory
  limit and Render restarted it. Extra jobs wait their turn ("queued").
- release_memory(): after a heavy job, hand freed memory back to the system
  (Python + glibc otherwise keep it, so the process only ever grows).
- rss_mb(): current memory use, for the logs.
"""

import ctypes
import gc
import os
import threading
from contextlib import contextmanager

HEAVY_JOBS = max(1, int(os.getenv("HEAVY_JOBS", "1")))
_slots = threading.BoundedSemaphore(HEAVY_JOBS)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc (e.g. macOS / Windows dev machines)
    _libc = None


def rss_mb() -> int:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def release_memory() -> None:
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


@contextmanager
def heavy_slot(label: str = "job"):
    if not _slots.acquire(blocking=False):
        print(f"[MEMORY] {label}: waiting for another document to finish (rss={rss_mb()} MB)")
        _slots.acquire()
    print(f"[MEMORY] {label}: start (rss={rss_mb()} MB)")
    try:
        yield
    finally:
        release_memory()
        print(f"[MEMORY] {label}: done (rss={rss_mb()} MB)")
        _slots.release()
