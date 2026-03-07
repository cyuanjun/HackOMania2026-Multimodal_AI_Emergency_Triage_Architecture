"""English translation adapter using the OpenAI chat API."""

from __future__ import annotations

import os

from app.config import AppConfig, DEFAULT_CONFIG
from app.env import ensure_dotenv_loaded


def translate_to_english(
    transcript_text: str,
    detected_language: str,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, str | None]:
    """Translate transcript text to English and preserve urgency."""

    if not transcript_text:
        return {"translated_text": "", "warning": None}

    language = (detected_language or "").lower()
    if language.startswith("en") or language == "english":
        return {"translated_text": transcript_text, "warning": None}

    ensure_dotenv_loaded()

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "translated_text": transcript_text,
            "warning": "translation: OPENAI_API_KEY not set",
        }

    try:
        from openai import OpenAI
    except Exception:
        return {
            "translated_text": transcript_text,
            "warning": "translation: openai package unavailable",
        }

    try:
        client = OpenAI()
        prompt = "Translate to English. Preserve urgency and emergency meaning. Text:\n" + transcript_text
        response = client.chat.completions.create(
            model=config.openai_translation_model,
            messages=[
                {"role": "system", "content": "You are a careful emergency-call translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        translated = response.choices[0].message.content.strip()
        return {"translated_text": translated or transcript_text, "warning": None}
    except Exception as exc:
        return {
            "translated_text": transcript_text,
            "warning": f"translation failed: {exc}",
        }
