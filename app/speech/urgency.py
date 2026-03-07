"""Voice strength and speech urgency scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import AppConfig, DEFAULT_CONFIG


def _clamp(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 3)


def compute_voice_strength(audio: np.ndarray) -> float:
    """Estimate voice strength from RMS and peak amplitude."""

    if audio.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))

    # Simple, transparent normalization tuned for 16k mono speech-like audio.
    rms_score = np.clip(rms / 0.12, 0.0, 1.0)
    peak_score = np.clip(peak / 0.9, 0.0, 1.0)
    return _clamp((0.65 * rms_score) + (0.35 * peak_score))


def compute_speech_urgency(
    keyword_features: dict[str, Any],
    voice_strength_score: float,
    non_speech_events: dict[str, Any],
    config: AppConfig = DEFAULT_CONFIG,
) -> float:
    """Combine keyword evidence and supporting acoustic cues."""

    help_signal = 1.0 if keyword_features.get("help_keyword_detected", False) else 0.0
    fall_signal = 1.0 if keyword_features.get("fall_keyword_detected", False) else 0.0
    cannot_breathe_signal = 1.0 if keyword_features.get("cannot_breathe_keyword_detected", False) else 0.0
    shouting_confidence = float(non_speech_events.get("shouting_confidence", 0.0))

    urgency = (
        (config.speech_urgency_help_weight * help_signal)
        + (config.speech_urgency_fall_weight * fall_signal)
        + (config.speech_urgency_cannot_breathe_weight * cannot_breathe_signal)
        + (config.speech_urgency_voice_strength_weight * float(voice_strength_score))
        + (config.speech_urgency_shouting_weight * shouting_confidence)
    )
    return _clamp(urgency)

