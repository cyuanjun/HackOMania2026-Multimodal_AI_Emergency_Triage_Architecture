"""Supplemental audio quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np


def assess_audio_quality(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return extra quality issues discovered after loading."""

    issues: list[str] = []
    if audio.size == 0:
        return {"quality_ok": False, "quality_issues": ["empty_audio"]}

    rms = float(np.sqrt(np.mean(np.square(audio))))
    dc_offset = float(abs(np.mean(audio)))
    if rms < 0.003:
        issues.append("very_low_signal")
    if dc_offset > 0.05:
        issues.append("dc_offset")
    if sample_rate < 8_000:
        issues.append("low_sample_rate")

    return {"quality_ok": len(issues) == 0, "quality_issues": issues}

