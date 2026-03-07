"""LLM-assisted explanation generation from structured audio signals."""

from __future__ import annotations

import json
import os
from typing import Any

from app.config import AppConfig, DEFAULT_CONFIG
from app.env import ensure_dotenv_loaded


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(number, 1.0))


def _build_explanation_context(result: dict[str, Any]) -> dict[str, Any]:
    """Select compact evidence for LLM explanation; no raw detection happens here."""

    non_speech = result.get("non_speech_events", {}) or {}
    speech = result.get("speech_features", {}) or {}
    transcript = result.get("transcript", {}) or {}
    language = result.get("language", {}) or {}
    meta = result.get("audio_meta", {}) or {}

    speech_urgency = _safe_float(speech.get("speech_urgency_score", 0.0))
    speech_keyword_strength = 1.0 if speech.get("keyword_hits") else 0.0
    non_speech_strength = max(
        _safe_float(non_speech.get("impact_confidence", 0.0)),
        _safe_float(non_speech.get("fall_sound_confidence", 0.0)),
        _safe_float(non_speech.get("shouting_confidence", 0.0)),
        _safe_float(non_speech.get("crying_confidence", 0.0)),
        _safe_float(non_speech.get("breathing_irregularity_confidence", 0.0)),
        _safe_float(non_speech.get("silence_after_impact_confidence", 0.0)),
    )
    speech_strength = max(speech_urgency, speech_keyword_strength)

    if speech_strength >= 0.6 and non_speech_strength >= 0.6:
        dominant = "fusion"
    elif speech_strength >= non_speech_strength:
        dominant = "speech"
    else:
        dominant = "non_speech"

    return {
        "audio_meta": {
            "quality_ok": bool(meta.get("quality_ok", True)),
            "quality_issues": list(meta.get("quality_issues", [])),
            "duration_sec": meta.get("duration_sec", 0.0),
        },
        "language": {
            "detected_language": language.get("detected_language", "unknown"),
            "confidence": _safe_float(language.get("confidence", 0.0)),
        },
        "transcript": {
            "text": transcript.get("text", ""),
            "translated_text": transcript.get("translated_text", ""),
        },
        "speech_features": {
            "help_keyword_detected": bool(speech.get("help_keyword_detected", False)),
            "fall_keyword_detected": bool(speech.get("fall_keyword_detected", False)),
            "cannot_breathe_keyword_detected": bool(speech.get("cannot_breathe_keyword_detected", False)),
            "keyword_hits": list(speech.get("keyword_hits", [])),
            "voice_strength_score": _safe_float(speech.get("voice_strength_score", 0.0)),
            "speech_urgency_score": speech_urgency,
        },
        "non_speech_events": {
            "crying_detected": bool(non_speech.get("crying_detected", False)),
            "crying_confidence": _safe_float(non_speech.get("crying_confidence", 0.0)),
            "shouting_detected": bool(non_speech.get("shouting_detected", False)),
            "shouting_confidence": _safe_float(non_speech.get("shouting_confidence", 0.0)),
            "impact_detected": bool(non_speech.get("impact_detected", False)),
            "impact_confidence": _safe_float(non_speech.get("impact_confidence", 0.0)),
            "fall_sound_detected": bool(non_speech.get("fall_sound_detected", False)),
            "fall_sound_confidence": _safe_float(non_speech.get("fall_sound_confidence", 0.0)),
            "breathing_irregularity_detected": bool(non_speech.get("breathing_irregularity_detected", False)),
            "breathing_irregularity_confidence": _safe_float(non_speech.get("breathing_irregularity_confidence", 0.0)),
            "silence_after_impact_detected": bool(non_speech.get("silence_after_impact_detected", False)),
            "silence_after_impact_confidence": _safe_float(non_speech.get("silence_after_impact_confidence", 0.0)),
        },
        "fusion_hints": {
            "speech_strength": round(speech_strength, 3),
            "non_speech_strength": round(non_speech_strength, 3),
            "dominant_evidence": dominant,
            "supports_fusion": speech_strength >= 0.6 and non_speech_strength >= 0.6,
        },
    }


def _fallback_explanations(
    context: dict[str, Any],
    fallback_explanations: list[str],
    max_items: int,
) -> dict[str, Any]:
    """Deterministic fallback when LLM is unavailable."""

    non_speech = context["non_speech_events"]
    speech = context["speech_features"]
    fusion = context["fusion_hints"]
    meta = context["audio_meta"]

    items: list[str] = []
    for line in fallback_explanations:
        if line and line not in items:
            items.append(line)

    if speech["keyword_hits"]:
        items.append(f"Transcript contains emergency-related terms: {', '.join(speech['keyword_hits'][:3])}.")
    elif speech["speech_urgency_score"] >= 0.5:
        items.append("Speech urgency appears elevated even with limited explicit keywords.")

    if speech["cannot_breathe_keyword_detected"]:
        items.append("Transcript may indicate possible respiratory distress.")
    if speech["fall_keyword_detected"] and (non_speech["impact_detected"] or non_speech["fall_sound_detected"]):
        items.append("Fall-related speech is consistent with impact/fall-like acoustic evidence.")

    if fusion["supports_fusion"]:
        items.append("Both speech and non-speech evidence suggest a potentially urgent incident.")
    elif fusion["dominant_evidence"] == "speech":
        items.append("Speech evidence appears stronger than acoustic non-speech evidence for this clip.")
    elif fusion["dominant_evidence"] == "non_speech":
        items.append("Acoustic non-speech evidence appears stronger than speech evidence for this clip.")

    if not meta["quality_ok"] and meta["quality_issues"]:
        items.append("Audio quality issues may reduce confidence in some signals.")

    unique = list(dict.fromkeys(items))
    if not unique:
        unique = ["No strong distress evidence was detected; weak signals may still warrant monitoring."]

    explanations = unique[: max(3, min(max_items, 5))]
    summary = explanations[0]
    return {"explanations": explanations, "summary": summary}


def _llm_messages(context: dict[str, Any], max_items: int) -> list[dict[str, str]]:
    system_prompt = (
        "You generate emergency-audio explanations for operators. "
        "Use only the provided structured analysis results. "
        "Do not invent evidence. "
        "If confidence is weak, use cautious wording such as 'possible', 'suggests', 'may indicate', or 'weakly supports'. "
        "Prefer the strongest evidence. "
        "Produce concise plain-English explanations. "
        "Return valid JSON only with keys: explanations, summary."
    )
    user_prompt = {
        "rules": {
            "max_explanations": max(3, min(max_items, 5)),
            "min_explanations": 3,
            "style": "operator-facing, concise, non-alarmist",
            "evidence_sources": ["non_speech", "speech", "fusion"],
            "do_not_include": ["new detections", "unsupported claims", "made-up details"],
        },
        "analysis": context,
        "output_schema": {
            "explanations": ["string", "string", "string"],
            "summary": "single sentence",
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
    ]


def _call_llm_for_explanations(context: dict[str, Any], config: AppConfig) -> dict[str, Any] | None:
    if not config.use_llm_explanations:
        return None

    ensure_dotenv_loaded()
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=config.openai_explanation_model,
            messages=_llm_messages(context, config.explanation_limit),
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        payload = json.loads(content)

        raw_explanations = payload.get("explanations", [])
        if not isinstance(raw_explanations, list):
            return None
        explanations = [str(item).strip() for item in raw_explanations if str(item).strip()]
        if not explanations:
            return None
        explanations = explanations[: max(3, min(config.explanation_limit, 5))]
        summary = str(payload.get("summary", "")).strip()
        return {"explanations": explanations, "summary": summary}
    except Exception:
        return None


def generate_explanations(
    result: dict[str, Any],
    config: AppConfig = DEFAULT_CONFIG,
    fallback_explanations: list[str] | None = None,
) -> dict[str, Any]:
    """Generate concise operator-facing explanations from already-computed signals."""

    fallback_list = list(fallback_explanations or [])
    context = _build_explanation_context(result)

    llm_payload = _call_llm_for_explanations(context, config)
    if llm_payload is not None:
        return llm_payload

    return _fallback_explanations(context, fallback_list, config.explanation_limit)

