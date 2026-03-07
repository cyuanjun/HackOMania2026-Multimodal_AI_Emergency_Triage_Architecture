"""End-to-end audio analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audio.event_detection import detect_events
from app.audio.loader import load_audio
from app.audio.quality import assess_audio_quality
from app.config import AppConfig, DEFAULT_CONFIG, VERSION
from app.explanations import generate_explanations
from app.speech.asr_openai import transcribe_audio_bytes
from app.speech.keywords import extract_speech_keywords
from app.speech.language import build_language_output
from app.speech.translation import translate_to_english
from app.speech.urgency import compute_speech_urgency, compute_voice_strength


def _empty_non_speech_output() -> dict[str, Any]:
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


def _empty_output() -> dict[str, Any]:
    output = _empty_non_speech_output()
    output["language"] = build_language_output()
    output["transcript"] = {
        "text": "",
        "translated_text": "",
        "asr_confidence": None,
    }
    output["speech_features"] = {
        "help_keyword_detected": False,
        "fall_keyword_detected": False,
        "cannot_breathe_keyword_detected": False,
        "keyword_hits": [],
        "voice_strength_score": 0.0,
        "speech_urgency_score": 0.0,
    }
    return output


def _error_output(issue: str, include_speech: bool = False) -> dict[str, Any]:
    output = _empty_output() if include_speech else _empty_non_speech_output()
    output["audio_meta"]["quality_issues"] = [issue]
    return output


def analyze_non_speech(audio_path: str | Path, config: AppConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Analyze an audio clip and return non-speech distress signals."""

    fallback = _empty_non_speech_output()
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


def analyze_audio(audio_path: str | Path, config: AppConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Analyze audio and extend the non-speech output with transcript fields."""

    try:
        loaded = load_audio(audio_path, config=config)
        audio = loaded["audio"]
        audio_meta = loaded["audio_meta"]

        extra_quality = assess_audio_quality(audio, audio_meta["sample_rate"])
        merged_issues = list(dict.fromkeys(audio_meta["quality_issues"] + extra_quality["quality_issues"]))
        audio_meta["quality_ok"] = len(merged_issues) == 0
        audio_meta["quality_issues"] = merged_issues

        detection = detect_events(audio, audio_meta["sample_rate"], config=config)

        transcript_info = transcribe_audio_bytes(
            loaded.get("audio_bytes", b""),
            loaded.get("source_path", Path(audio_path)).name,
            config=config,
        )
        translation_info = translate_to_english(
            transcript_info["text"],
            transcript_info["detected_language"],
            config=config,
        )
        try:
            keyword_features = extract_speech_keywords(
                transcript_info["text"],
                translation_info["translated_text"],
                config=config,
            )
        except Exception:
            keyword_features = {
                "help_keyword_detected": False,
                "fall_keyword_detected": False,
                "cannot_breathe_keyword_detected": False,
                "keyword_hits": [],
            }

        try:
            voice_strength_score = compute_voice_strength(audio)
        except Exception:
            voice_strength_score = 0.0

        try:
            speech_urgency_score = compute_speech_urgency(
                keyword_features,
                voice_strength_score,
                detection["non_speech_events"],
                config=config,
            )
        except Exception:
            speech_urgency_score = 0.0

        warning_messages = [
            f"speech_warning:{warning}"
            if not str(warning).startswith("speech_warning:")
            else str(warning)
            for warning in (transcript_info.get("warning"), translation_info.get("warning"))
            if warning
        ]
        if warning_messages:
            audio_meta["quality_issues"] = list(dict.fromkeys(audio_meta["quality_issues"] + warning_messages))

        result = {
            "audio_meta": audio_meta,
            "non_speech_events": detection["non_speech_events"],
            "acoustic_features": detection["acoustic_features"],
            "language": build_language_output(
                transcript_info["detected_language"],
                transcript_info["language_confidence"],
            ),
            "transcript": {
                "text": transcript_info["text"],
                "translated_text": translation_info["translated_text"],
                "asr_confidence": transcript_info["asr_confidence"],
            },
            "speech_features": {
                "help_keyword_detected": bool(keyword_features["help_keyword_detected"]),
                "fall_keyword_detected": bool(keyword_features["fall_keyword_detected"]),
                "cannot_breathe_keyword_detected": bool(keyword_features["cannot_breathe_keyword_detected"]),
                "keyword_hits": list(keyword_features["keyword_hits"]),
                "voice_strength_score": float(voice_strength_score),
                "speech_urgency_score": float(speech_urgency_score),
            },
            "explanations": detection["explanations"],
            "model_info": {
                "event_model": detection["event_model"],
                "version": VERSION,
            },
        }
        explanation_payload = generate_explanations(
            result,
            config=config,
            fallback_explanations=detection["explanations"],
        )
        result["explanations"] = list(explanation_payload.get("explanations", result["explanations"]))
        return result
    except FileNotFoundError as exc:
        return _error_output(f"file_not_found:{exc}", include_speech=True)
    except ValueError as exc:
        return _error_output(f"audio_decode_failed:{exc}", include_speech=True)
    except Exception as exc:
        return _error_output(f"analysis_failed:{type(exc).__name__}", include_speech=True)
