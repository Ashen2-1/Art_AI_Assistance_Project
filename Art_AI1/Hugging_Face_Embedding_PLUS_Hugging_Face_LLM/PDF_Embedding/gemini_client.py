import os
from pathlib import Path
from typing import Optional
import base64
import mimetypes

import requests
from dotenv import load_dotenv


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


class GeminiAPIError(RuntimeError):
    """Gemini API request failed."""


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
    """

    if not GEMINI_API_KEY:
        raise GeminiAPIError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

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
        response = requests.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json=payload,
            timeout=90,
        )
    except requests.Timeout as error:
        raise GeminiAPIError(
            "Gemini request timed out. Please try again."
        ) from error
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

    if not GEMINI_API_KEY:
        raise GeminiAPIError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}"
        )

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image was not found: {image_path}"
        )

    mime_type, _ = mimetypes.guess_type(image_file.name)

    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded_image = base64.b64encode(
        image_file.read_bytes()
    ).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded_image,
                        }
                    },
                    {
                        "text": prompt.strip(),
                    },
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
        response = requests.post(
            GEMINI_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json=payload,
            timeout=120,
        )
    except requests.Timeout as error:
        raise GeminiAPIError(
            "Gemini image request timed out."
        ) from error
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