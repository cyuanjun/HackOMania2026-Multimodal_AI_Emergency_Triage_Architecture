"""End-to-end non-speech audio analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audio.event_detection import detect_events
from app.audio.loader import load_audio
from app.audio.quality import assess_audio_quality
from app.config import AppConfig, DEFAULT_CONFIG, VERSION


def _empty_output() -> dict[str, Any]:
    return {
        "audio_meta": {
            "duration_sec": 0.0,
            "sample_rate": DEFAULT_CONFIG.target_sample_rate,
            "channels": 1,
            "quality_ok": False,
            "quality_issues": ["analysis_failed"],
        },
        "non_speech_events": {
            "crying_detected": False,
            "crying_confidence": 0.0,
            "shouting_detected": False,
            "shouting_confidence": 0.0,
            "impact_detected": False,
            "impact_confidence": 0.0,
            "fall_sound_detected": False,
            "fall_sound_confidence": 0.0,
            "breathing_irregularity_detected": False,
            "breathing_irregularity_confidence": 0.0,
            "silence_after_impact_detected": False,
            "silence_after_impact_confidence": 0.0,
        },
        "acoustic_features": {
            "rms_energy_mean": 0.0,
            "rms_energy_max": 0.0,
            "silence_ratio": 1.0,
            "peak_count": 0,
            "sudden_impact_score": 0.0,
            "breathing_variability_score": 0.0,
        },
        "explanations": [],
        "model_info": {
            "event_model": "heuristic_refinement_v2",
            "version": VERSION,
        },
    }


def _error_output(issue: str) -> dict[str, Any]:
    output = _empty_output()
    output["audio_meta"]["quality_issues"] = [issue]
    return output


def analyze_non_speech(audio_path: str | Path, config: AppConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Analyze an audio clip and return non-speech distress signals."""

    fallback = _empty_output()
    try:
        loaded = load_audio(audio_path, config=config)
        audio = loaded["audio"]
        audio_meta = loaded["audio_meta"]

        extra_quality = assess_audio_quality(audio, audio_meta["sample_rate"])
        merged_issues = list(dict.fromkeys(audio_meta["quality_issues"] + extra_quality["quality_issues"]))
        audio_meta["quality_ok"] = len(merged_issues) == 0
        audio_meta["quality_issues"] = merged_issues

        detection = detect_events(audio, audio_meta["sample_rate"], config=config)
        return {
            "audio_meta": audio_meta,
            "non_speech_events": detection["non_speech_events"],
            "acoustic_features": detection["acoustic_features"],
            "explanations": detection["explanations"],
            "model_info": {
                "event_model": detection["event_model"],
                "version": VERSION,
            },
        }
    except FileNotFoundError as exc:
        return _error_output(f"file_not_found:{exc}")
    except ValueError as exc:
        return _error_output(f"audio_decode_failed:{exc}")
    except Exception as exc:
        return _error_output(f"analysis_failed:{type(exc).__name__}")
