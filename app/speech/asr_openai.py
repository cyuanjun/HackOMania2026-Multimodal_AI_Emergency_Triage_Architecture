"""OpenAI Whisper API adapter for transcript extraction."""

from __future__ import annotations

import io
import os
from typing import Any

from app.config import AppConfig, DEFAULT_CONFIG
from app.env import ensure_dotenv_loaded


def transcribe_audio_bytes(
    audio_bytes: bytes,
    audio_filename: str,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Transcribe audio with the OpenAI audio API.

    This mirrors the external `pab_audio` module structure: use Whisper API
    for transcript + language, and fail open when the API or key is unavailable.
    """

    default_result = {
        "text": "",
        "asr_confidence": None,
        "detected_language": "unknown",
        "language_confidence": 0.0,
        "warning": None,
    }

    if not audio_bytes:
        default_result["warning"] = "asr_openai: no audio provided"
        return default_result

    ensure_dotenv_loaded()

    if not os.getenv("OPENAI_API_KEY"):
        default_result["warning"] = "asr_openai: OPENAI_API_KEY not set"
        return default_result

    try:
        from openai import OpenAI
    except Exception:
        default_result["warning"] = "asr_openai: openai package unavailable"
        return default_result

    try:
        client = OpenAI()
        buffer = io.BytesIO(audio_bytes)
        buffer.name = audio_filename or "audio.wav"

        response = client.audio.transcriptions.create(
            model=config.openai_whisper_model,
            file=buffer,
            response_format="verbose_json",
        )
        text = getattr(response, "text", "") or ""
        language = getattr(response, "language", None) or "unknown"

        segment_confidences: list[float] = []
        for segment in getattr(response, "segments", []) or []:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None:
                try:
                    # Whisper returns logprob, so convert it to a bounded proxy.
                    segment_confidences.append(max(0.0, min(1.0, float(pow(2.718281828, avg_logprob)))))
                except Exception:
                    continue

        asr_confidence = round(sum(segment_confidences) / len(segment_confidences), 3) if segment_confidences else None

        return {
            "text": text,
            "asr_confidence": asr_confidence,
            "detected_language": language,
            "language_confidence": 0.99 if language != "unknown" and text else 0.0,
            "warning": None,
        }
    except Exception as exc:
        default_result["warning"] = f"asr_openai failed: {exc}"
        return default_result
