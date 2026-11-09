import os
from pathlib import Path
from typing import Optional
import base64
import io
import mimetypes

import re
import time
import requests
from dotenv import load_dotenv

import user_keys


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()

# Photo work (Photo Detective, image cards) needs more than the lite model.
# A model name that does not exist falls back to GEMINI_MODEL (404 handling).
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest").strip() or GEMINI_MODEL

# Images are shrunk before upload: big scans are slow to send and to read,
# and 1600 px is plenty to compare faces, clothes and studio backdrops.
IMAGE_MAX_SIDE = int(os.getenv("GEMINI_IMAGE_MAX_SIDE", "1600"))

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


class GeminiAPIError(RuntimeError):
    """Gemini API request failed."""


class UserKeyError(GeminiAPIError):
    """The user's own API key was rejected or their provider failed."""


PROVIDER_NAMES = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Claude (Anthropic)",
    "groq": "Groq (Llama)",
    "qwen": "Qwen (Alibaba Cloud)",
}

# Free / low-cost providers with an OpenAI-compatible chat API.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _gemini_url(model: str = "") -> str:
    return (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model or GEMINI_MODEL}:generateContent"
    )


GEMINI_BUSY_CODES = {429, 500, 503, 504}
GEMINI_RETRY_DELAYS = (3, 8, 15)


def _post_gemini(model: str, api_key: str, payload: dict, timeout: int) -> requests.Response:
    """
    POST to Gemini with two safety nets:
    - busy (429/500/503/504, e.g. "model is experiencing high demand"):
      wait and retry a few times, switching to NEXO's default model for
      the later tries when another model was chosen;
    - retired model (404, e.g. an old model picked in Settings -> AI keys):
      retry with NEXO's default model;
    - daily quota of the chosen model used up (429 "quota exceeded", e.g. a
      free-tier key allows only ~20 requests a day on a big model): try the
      default lite model once, which has its own, larger quota.
    """
    models = [model or GEMINI_MODEL]
    if models[0] != GEMINI_MODEL:
        models.append(GEMINI_MODEL)

    def send(name):
        return requests.post(
            _gemini_url(name),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=timeout,
        )

    current = models[0]
    response = send(current)
    if response.status_code == 404 and len(models) > 1:
        print(f"[Gemini] model {current!r} unavailable (404); using {GEMINI_MODEL!r}")
        current = GEMINI_MODEL
        response = send(current)

    for attempt, delay in enumerate(GEMINI_RETRY_DELAYS, start=1):
        if response.status_code not in GEMINI_BUSY_CODES:
            break
        # 429 from a quota that is used up will not clear in seconds.
        if response.status_code == 429 and "quota" in (response.text or "").lower() and "exceeded" in (response.text or "").lower():
            if current != GEMINI_MODEL:
                print(f"[Gemini] quota for {current!r} used up; trying {GEMINI_MODEL!r}")
                current = GEMINI_MODEL
                response = send(current)
                continue
            break
        if attempt >= 2 and current != GEMINI_MODEL:
            current = GEMINI_MODEL
        print(f"[Gemini] busy ({response.status_code}); retry {attempt}/{len(GEMINI_RETRY_DELAYS)} in {delay}s with {current!r}")
        time.sleep(delay)
        response = send(current)
    return response


def _post_gemini_patient(model: str, api_key: str, payload: dict, timeout: int, label: str = "request") -> requests.Response:
    """
    _post_gemini, but a model that is too slow (e.g. a Pro model thinking
    through several photos) gets one more try on the fast default model
    instead of failing with a timeout.
    """
    try:
        return _post_gemini(model, api_key, payload, timeout)
    except requests.Timeout as error:
        slow = model or GEMINI_MODEL
        if slow != GEMINI_MODEL:
            print(f"[Gemini] {label} on {slow!r} timed out after {timeout}s; retrying with {GEMINI_MODEL!r}")
            try:
                return _post_gemini(GEMINI_MODEL, api_key, payload, max(60, timeout - 60))
            except requests.Timeout:
                pass
        raise GeminiAPIError(
            f"Gemini took too long to answer ({slow}). Please try again"
            + (" — or choose a Flash model in Settings → AI keys, which is much faster with photos." if "pro" in slow.lower() else ".")
        ) from error


def _image_part(image_path) -> dict:
    """Inline image for Gemini, shrunk to IMAGE_MAX_SIDE and re-encoded as JPEG."""
    image_file = Path(image_path)
    raw = image_file.read_bytes()
    mime_type, _ = mimetypes.guess_type(image_file.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw)) as picture:
            picture = ImageOps.exif_transpose(picture)
            if max(picture.size) > IMAGE_MAX_SIDE or len(raw) > 1_500_000 or mime_type not in ("image/jpeg", "image/png", "image/webp"):
                picture = picture.convert("RGB")
                picture.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=86, optimize=True)
                raw = buffer.getvalue()
                mime_type = "image/jpeg"
    except Exception as error:  # unreadable by Pillow: send the original
        print(f"[Gemini] could not shrink {image_file.name}: {error}")
    return {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(raw).decode("utf-8")}}


def _raise_user_provider_error(provider: str, response: requests.Response) -> None:
    name = PROVIDER_NAMES.get(provider, provider)
    message = _read_error_message(response)
    if response.status_code in (401, 403):
        raise UserKeyError(
            f"Your {name} API key was rejected ({response.status_code}). "
            "Check it in Settings → AI keys, or switch back to the NEXO default."
        )
    if response.status_code == 402:
        raise UserKeyError(
            f"Your {name} account has no prepaid credit left, so it cannot answer. "
            + ("Add credit to that project's billing (Google AI Studio → Projects → Billing), " if provider == "gemini" else "Add credit with that provider, ")
            + "or open Settings → AI keys and remove your key to use the NEXO default for now."
        )
    if response.status_code == 429:
        raise UserKeyError(
            f"Your {name} account hit its rate limit or has no credit left. "
            f"Details: {message}"
        )
    raise UserKeyError(f"{name} returned HTTP {response.status_code}: {message}")


def _generate_openai(prompt, system_instruction, temperature, max_output_tokens, key, model):
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        "messages": messages,
        # Reasoning models spend part of the budget thinking.
        "max_completion_tokens": max(max_output_tokens * 2, 4096),
        "temperature": temperature,
    }

    for attempt in range(2):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
        except requests.RequestException as error:
            raise UserKeyError(f"Could not reach OpenAI: {error}") from error

        # Some models only accept the default temperature.
        if response.status_code == 400 and attempt == 0 and "temperature" in response.text:
            payload.pop("temperature", None)
            continue
        break

    if not response.ok:
        _raise_user_provider_error("openai", response)

    data = response.json()
    choices = data.get("choices") or []
    answer = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not answer:
        raise UserKeyError("OpenAI returned an empty answer.")
    return answer


def _generate_compatible(provider, base_url, prompt, system_instruction, temperature, max_output_tokens, key, model):
    """Groq (Llama), Qwen and other OpenAI-compatible chat APIs."""
    name = PROVIDER_NAMES.get(provider, provider)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max(max_output_tokens, 1024),
        "temperature": temperature,
    }

    response = None
    for attempt in range(3):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
        except requests.RequestException as error:
            raise UserKeyError(f"Could not reach {name}: {error}") from error

        # Free tiers limit tokens per minute: wait briefly once or twice.
        if response.status_code == 429 and attempt < 2 and "day" not in (response.text or "").lower():
            retry_after = response.headers.get("retry-after") or ""
            try:
                delay = min(20.0, max(1.0, float(retry_after)))
            except ValueError:
                delay = 6.0 * (attempt + 1)
            print(f"[{name}] rate limited; retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        # A request bigger than the model allows: ask for a shorter answer once.
        if response.status_code in (400, 413) and attempt == 0 and "token" in (response.text or "").lower():
            payload["max_tokens"] = 1024
            continue
        break

    if not response.ok:
        if response.status_code == 413 or (response.status_code == 400 and "context" in (response.text or "").lower()):
            raise UserKeyError(
                f"The question with its sources is too long for this {name} model. "
                "Select fewer cards, or choose a model with a larger context in Settings → AI keys."
            )
        _raise_user_provider_error(provider, response)

    data = response.json()
    choices = data.get("choices") or []
    answer = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    # Reasoning models (e.g. Qwen3) may wrap their thinking in <think> tags.
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
    if not answer:
        raise UserKeyError(f"{name} returned an empty answer.")
    return answer


def _generate_anthropic(prompt, system_instruction, temperature, max_output_tokens, key, model):
    payload = {
        "model": model or os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-5"),
        "max_tokens": max(max_output_tokens, 2048),
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt.strip()}],
    }
    if system_instruction:
        payload["system"] = system_instruction.strip()

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
    except requests.RequestException as error:
        raise UserKeyError(f"Could not reach Anthropic: {error}") from error

    if not response.ok:
        _raise_user_provider_error("anthropic", response)

    data = response.json()
    answer = "\n".join(
        block.get("text", "") for block in data.get("content") or [] if block.get("type") == "text"
    ).strip()
    if not answer:
        raise UserKeyError("Claude returned an empty answer.")
    return answer


def _read_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
        return (
            data.get("error", {}).get("message")
            or data.get("message")
            or response.text
        )
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def _extract_answer(data: dict) -> str:
    candidates = data.get("candidates") or []

    if not candidates:
        prompt_feedback = data.get("promptFeedback") or {}
        block_reason = prompt_feedback.get("blockReason")

        if block_reason:
            raise GeminiAPIError(
                f"Gemini blocked the request: {block_reason}"
            )

        raise GeminiAPIError(
            "Gemini returned no answer candidates."
        )

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []

    text_parts = [
        part.get("text", "").strip()
        for part in parts
        if part.get("text")
    ]

    answer = "\n".join(
        text for text in text_parts if text
    ).strip()

    if not answer:
        finish_reason = candidates[0].get(
            "finishReason",
            "unknown",
        )

        raise GeminiAPIError(
            "Gemini returned an empty answer. "
            f"Finish reason: {finish_reason}"
        )

    return answer


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    """
    Send a text request to Gemini and return the generated answer.

    The API key is sent through a request header, so it will not
    appear inside the URL or normal error output.

    If the user connected their own key (Settings → AI keys), the answer
    comes from their provider: OpenAI, Claude, or Gemini on their key.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    provider, user_key, user_model = user_keys.answer_settings()

    if provider == "openai" and user_key:
        return _generate_openai(prompt, system_instruction, temperature, max_output_tokens, user_key, user_model)

    if provider == "anthropic" and user_key:
        return _generate_anthropic(prompt, system_instruction, temperature, max_output_tokens, user_key, user_model)

    if provider == "groq" and user_key:
        return _generate_compatible(
            "groq", GROQ_BASE_URL, prompt, system_instruction, temperature, max_output_tokens, user_key,
            user_model or os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile"),
        )

    if provider == "qwen" and user_key:
        return _generate_compatible(
            "qwen", user_keys.qwen_base_url(user_keys.answer_endpoint()), prompt, system_instruction, temperature,
            max_output_tokens, user_key, user_model or os.getenv("QWEN_DEFAULT_MODEL", "qwen-plus"),
        )

    own_gemini = provider == "gemini" and bool(user_key)
    api_key = user_key if own_gemini else GEMINI_API_KEY
    gemini_model = user_model if own_gemini and user_model else GEMINI_MODEL

    if not api_key:
        raise GeminiAPIError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt.strip(),
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [
                {
                    "text": system_instruction.strip(),
                }
            ]
        }

    try:
        # Long answers (reports, Photo Detective reasoning) need more time.
        response = _post_gemini_patient(gemini_model, api_key, payload, 90 if max_output_tokens <= 2048 else 150, "answer")
    except requests.RequestException as error:
        raise GeminiAPIError(
            "Could not connect to the Gemini API."
        ) from error

    if not response.ok:
        if own_gemini:
            _raise_user_provider_error("gemini", response)

        message = _read_error_message(response)

        raise GeminiAPIError(
            f"Gemini API returned HTTP "
            f"{response.status_code}: {message}"
        )

    return _extract_answer(response.json())

def generate_image(
    prompt: str,
    image_path: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    """
    Send an image and text prompt to Gemini.
    """

    image_key = user_keys.gemini_key(GEMINI_API_KEY)

    if not image_key:
        raise GeminiAPIError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image was not found: {image_path}"
        )

    payload = {
        "contents": [{"role": "user", "parts": [_image_part(image_file), {"text": prompt.strip()}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}

    try:
        response = _post_gemini_patient(GEMINI_VISION_MODEL, image_key, payload, 120, "image description")
    except requests.RequestException as error:
        raise GeminiAPIError(
            "Could not connect to the Gemini API."
        ) from error

    if not response.ok:
        message = _read_error_message(response)

        raise GeminiAPIError(
            f"Gemini API returned HTTP "
            f"{response.status_code}: {message}"
        )

    return _extract_answer(response.json())


def generate_multimodal(
    prompt: str,
    image_paths: list,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    json_output: bool = False,
) -> str:
    """
    One Gemini request with several images (e.g. two photographs to
    compare). Uses the user's own Gemini key when connected
    (Settings -> AI keys), otherwise NEXO's key.
    """

    provider, user_key, user_model = user_keys.answer_settings()
    api_key = user_keys.gemini_key(GEMINI_API_KEY)
    model = user_model if provider == "gemini" and user_model else GEMINI_VISION_MODEL
    own_key = user_keys.using_own_gemini_key()

    if not api_key:
        raise GeminiAPIError(f"GEMINI_API_KEY was not found in {ENV_PATH}")

    parts = [_image_part(image_path) for image_path in image_paths]
    parts.append({"text": prompt.strip()})

    generation_config = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if json_output:
        generation_config["responseMimeType"] = "application/json"

    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}

    try:
        response = _post_gemini_patient(model, api_key, payload, 150, "image request")
    except requests.RequestException as error:
        raise GeminiAPIError("Could not connect to the Gemini API.") from error

    if not response.ok:
        if own_key:
            _raise_user_provider_error("gemini", response)
        raise GeminiAPIError(f"Gemini API returned HTTP {response.status_code}: {_read_error_message(response)}")

    return _extract_answer(response.json())
