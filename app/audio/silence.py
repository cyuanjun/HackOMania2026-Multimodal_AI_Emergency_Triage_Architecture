"""Silence analysis helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _detect_silence_regions_librosa(audio: np.ndarray, sample_rate: int, top_db: float) -> dict[str, Any]:
    """Prefer librosa when available so the silence path stays aligned with the original stack."""

    import librosa

    intervals = librosa.effects.split(audio, top_db=top_db)
    total_duration = len(audio) / sample_rate
    if len(intervals) == 0:
        return {"silence_ratio": 1.0, "regions": [(0.0, total_duration)]}

    regions: list[tuple[float, float]] = []
    last_end = 0
    for start, end in intervals:
        if start > last_end:
            regions.append((last_end / sample_rate, start / sample_rate))
        last_end = end
    if last_end < len(audio):
        regions.append((last_end / sample_rate, len(audio) / sample_rate))

    silent_time = sum(max(end - start, 0.0) for start, end in regions)
    return {
        "silence_ratio": round(float(silent_time / max(total_duration, 1e-6)), 3),
        "regions": regions,
    }


def _detect_silence_regions_rms(audio: np.ndarray, sample_rate: int, top_db: float) -> dict[str, Any]:
    """Fallback silence detector using frame RMS when librosa is unavailable."""

    frame_length = min(2048, max(256, int(sample_rate * 0.064)))
    hop_length = max(128, frame_length // 4)
    pad = frame_length // 2
    padded = np.pad(audio, (pad, pad), mode="constant")

    frames = []
    for start in range(0, len(padded) - frame_length + 1, hop_length):
        frame = padded[start : start + frame_length]
        frames.append(float(np.sqrt(np.mean(np.square(frame)))))

    rms = np.asarray(frames, dtype=np.float32)
    if rms.size == 0:
        return {"silence_ratio": 1.0, "regions": [(0.0, len(audio) / sample_rate)]}

    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    threshold = float(np.max(db) - top_db)
    silent_mask = db < threshold

    regions: list[tuple[float, float]] = []
    active_start: int | None = None
    for idx, is_silent in enumerate(silent_mask):
        if is_silent and active_start is None:
            active_start = idx
        elif not is_silent and active_start is not None:
            start_sec = active_start * hop_length / sample_rate
            end_sec = idx * hop_length / sample_rate
            regions.append((start_sec, end_sec))
            active_start = None
    if active_start is not None:
        regions.append((active_start * hop_length / sample_rate, len(audio) / sample_rate))

    silent_time = sum(max(end - start, 0.0) for start, end in regions)
    return {
        "silence_ratio": round(float(silent_time / max(len(audio) / sample_rate, 1e-6)), 3),
        "regions": regions,
    }


def detect_silence_regions(audio: np.ndarray, sample_rate: int, top_db: float) -> dict[str, Any]:
    """Estimate silence spans using librosa first, then a frame-RMS fallback."""

    if audio.size == 0:
        return {"silence_ratio": 1.0, "regions": [(0.0, 0.0)]}

    try:
        return _detect_silence_regions_librosa(audio, sample_rate, top_db)
    except Exception:
        return _detect_silence_regions_rms(audio, sample_rate, top_db)


def detect_silence_after_impact(
    audio: np.ndarray,
    sample_rate: int,
    impact_index: int | None,
    top_db: float,
    min_silence_sec: float,
    impact_confidence: float = 1.0,
    impact_gate_threshold: float = 0.55,
) -> tuple[bool, float]:
    """Score silence after a candidate impact location using post-impact RMS logic."""

    if impact_index is None or audio.size == 0 or impact_confidence < impact_gate_threshold:
        return False, 0.0

    post_audio = audio[max(impact_index, 0) :]
    if post_audio.size == 0:
        return False, 0.0

    silence = detect_silence_regions(post_audio, sample_rate, top_db=top_db)
    post_impact_silence_ratio = float(silence["silence_ratio"])

    pre_window = audio[max(0, impact_index - sample_rate // 2) : impact_index]
    post_window = post_audio[: max(int(sample_rate * min_silence_sec), 1)]
    if pre_window.size == 0:
        pre_window = audio[: max(int(sample_rate * 0.25), 1)]

    pre_rms = float(np.sqrt(np.mean(np.square(pre_window)))) if pre_window.size else 0.0
    post_rms = float(np.sqrt(np.mean(np.square(post_window)))) if post_window.size else 0.0
    rms_drop_score = float(np.clip((pre_rms - post_rms) / max(pre_rms, 1e-6), 0.0, 1.0))

    confidence = np.clip((0.6 * post_impact_silence_ratio) + (0.4 * rms_drop_score), 0.0, 1.0)
    return bool(confidence >= 0.45), round(float(confidence), 3)
