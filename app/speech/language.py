"""Language metadata normalization helpers."""

from __future__ import annotations


def build_language_output(
    detected_language: str = "unknown",
    confidence: float = 0.0,
) -> dict[str, object]:
    return {
        "detected_language": detected_language or "unknown",
        "detected_dialect": "unknown",
        "confidence": round(float(max(0.0, min(confidence, 1.0))), 3),
    }
