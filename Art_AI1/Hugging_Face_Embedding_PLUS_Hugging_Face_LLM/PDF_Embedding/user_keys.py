"""
Per-request "bring your own key" settings.

Express (the only caller; protected by the internal API key) may send:

    X-User-Gemini-Key   the user's Gemini key: used for embeddings
                        (indexing + search) and for Gemini answers
    X-Answer-Provider   nexo | gemini | openai | anthropic
    X-Answer-Key        key for the answer provider (openai / anthropic)
    X-Answer-Model      model id chosen by the user (optional)

Without headers everything uses NEXO's own GEMINI_API_KEY, as before.
Embeddings always use the same Gemini embedding model, so vectors made
with a user's key and with ours stay compatible.

The values live in a ContextVar for the duration of one request. Code
that runs in a worker thread must enter `using(...)` inside that thread.
"""

from contextlib import contextmanager
from contextvars import ContextVar

PROVIDERS = {"nexo", "gemini", "openai", "anthropic"}

_current: ContextVar = ContextVar("nexo_user_keys", default=None)


def _clean(value, limit=400):
    text = str(value or "").strip()
    return text[:limit] if text else ""


def from_headers(headers) -> dict:
    provider = _clean(headers.get("x-answer-provider"), 20).lower() or "nexo"
    if provider not in PROVIDERS:
        provider = "nexo"
    return {
        "gemini_key": _clean(headers.get("x-user-gemini-key")),
        "answer_provider": provider,
        "answer_key": _clean(headers.get("x-answer-key")),
        "answer_model": _clean(headers.get("x-answer-model"), 120),
    }


@contextmanager
def using(keys: dict):
    token = _current.set(keys or None)
    try:
        yield
    finally:
        _current.reset(token)


def _get() -> dict:
    return _current.get() or {}


def gemini_key(default: str = "") -> str:
    """The user's Gemini key if they connected one, else NEXO's."""
    return _get().get("gemini_key") or default


def using_own_gemini_key() -> bool:
    return bool(_get().get("gemini_key"))


def answer_settings():
    """(provider, key, model) for generating answers."""
    keys = _get()
    provider = keys.get("answer_provider") or "nexo"
    if provider == "gemini":
        return provider, keys.get("gemini_key", ""), keys.get("answer_model", "")
    if provider in ("openai", "anthropic"):
        return provider, keys.get("answer_key", ""), keys.get("answer_model", "")
    return "nexo", "", ""
