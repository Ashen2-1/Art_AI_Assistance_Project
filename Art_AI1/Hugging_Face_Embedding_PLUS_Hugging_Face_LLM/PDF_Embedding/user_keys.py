"""
Per-request "bring your own key" settings.

Express (the only caller; protected by the internal API key) may send:

    X-User-Gemini-Key   the user's Gemini key: used for embeddings
                        (indexing + search) and for Gemini answers
    X-Answer-Provider   nexo | gemini | openai | anthropic | groq | qwen | deepseek
    X-Answer-Key        key for the answer provider (all but gemini)
    X-Answer-Model      model id chosen by the user (optional)
    X-Answer-Endpoint   qwen only: region code, e.g. "intl", "us", "cn"
                        or "intl:<workspace id>" (never a URL; see
                        qwen_base_url, so a user cannot point us anywhere)

Without headers everything uses NEXO's own GEMINI_API_KEY, as before.
Embeddings always use the same Gemini embedding model, so vectors made
with a user's key and with ours stay compatible.

The values live in a ContextVar for the duration of one request. Code
that runs in a worker thread must enter `using(...)` inside that thread.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import re

PROVIDERS = {"nexo", "gemini", "openai", "anthropic", "groq", "qwen", "deepseek"}
KEYED_PROVIDERS = ("openai", "anthropic", "groq", "qwen", "deepseek")

# Alibaba Cloud Model Studio (Qwen), OpenAI-compatible mode. Keys are bound
# to the region they were created in.
_QWEN_HOSTS = {"intl": "dashscope-intl.aliyuncs.com", "us": "dashscope-us.aliyuncs.com", "cn": "dashscope.aliyuncs.com"}
_QWEN_WORKSPACE_REGIONS = {"intl": "ap-southeast-1", "cn": "cn-beijing"}
_ENDPOINT_PATTERN = re.compile(r"^(intl|us|cn)(?::([a-z0-9][a-z0-9-]{2,62}))?$")


def qwen_base_url(code: str) -> str:
    """Region code → base URL. Unknown codes fall back to the international endpoint."""
    match = _ENDPOINT_PATTERN.match(str(code or "").strip().lower())
    if not match:
        return f"https://{_QWEN_HOSTS['intl']}/compatible-mode/v1"
    region, workspace = match.group(1), match.group(2)
    if workspace and region in _QWEN_WORKSPACE_REGIONS:
        return f"https://{workspace}.{_QWEN_WORKSPACE_REGIONS[region]}.maas.aliyuncs.com/compatible-mode/v1"
    return f"https://{_QWEN_HOSTS[region]}/compatible-mode/v1"

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
        "answer_endpoint": _clean(headers.get("x-answer-endpoint"), 80),
        # "heavy" = AI reports, Photo Detective: NEXO uses its strong model.
        "task_tier": "heavy" if _clean(headers.get("x-task-tier"), 10).lower() == "heavy" else "",
        # Express sends "off" when NEXO's monthly Claude budget is used up.
        "claude_off": _clean(headers.get("x-nexo-claude"), 10).lower() == "off",
        # Every AI call made for this request (tokens, model, who pays): sent back to Express.
        "usage": [],
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
    if provider in KEYED_PROVIDERS:
        return provider, keys.get("answer_key", ""), keys.get("answer_model", "")
    return "nexo", "", ""


def claude_allowed() -> bool:
    return not _get().get("claude_off")


def record_usage(provider: str, model: str, input_tokens, output_tokens, payer: str, kind: str = "chat") -> None:
    """Remember one AI call for the usage dashboard (no-op outside a request)."""
    keys = _current.get()
    if keys is None:
        return
    try:
        entry = {
            "provider": provider,
            "model": str(model or "")[:80],
            "input": max(0, int(input_tokens or 0)),
            "output": max(0, int(output_tokens or 0)),
            "payer": "nexo" if payer == "nexo" else "user",
            "kind": kind,
        }
    except (TypeError, ValueError):
        return
    usage = keys.setdefault("usage", [])
    # Embedding batches: merge into one line per model.
    if kind == "embed":
        for item in usage:
            if item["kind"] == "embed" and item["model"] == entry["model"] and item["payer"] == entry["payer"]:
                item["input"] += entry["input"]
                return
    if len(usage) < 200:
        usage.append(entry)


def heavy_task() -> bool:
    return _get().get("task_tier") == "heavy"


def answer_endpoint() -> str:
    """Region code for providers that have one (Qwen)."""
    return _get().get("answer_endpoint", "")
