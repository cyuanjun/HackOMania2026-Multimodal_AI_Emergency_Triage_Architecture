"""Keyword extraction for emergency speech cues."""

from __future__ import annotations

import re
from typing import Any

from app.config import AppConfig, DEFAULT_CONFIG


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return " ".join(cleaned.split())


def _find_hits(text: str, patterns: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    tokens = set(text.split())
    for pattern in patterns:
        normalized_pattern = _normalize_text(pattern)
        if not normalized_pattern:
            continue
        # Keep simple matching, but avoid obvious false positives like "help" in "helping".
        if " " not in normalized_pattern:
            if normalized_pattern in tokens:
                hits.append(pattern)
            continue
        if normalized_pattern in text:
            hits.append(pattern)
    return hits


def _find_core_regex_hits(text: str, config: AppConfig) -> dict[str, list[str]]:
    """Conservative regex layer for high-value emergency phrase variants."""

    grouped_hits: dict[str, list[str]] = {"help": [], "fall": [], "cannot_breathe": []}
    for category, patterns in config.speech_core_regex_patterns.items():
        for label, pattern in patterns:
            if re.search(pattern, text):
                grouped_hits.setdefault(category, []).append(label)
    return grouped_hits


def extract_speech_keywords(
    transcript_text: str,
    translated_text: str,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Detect emergency keywords from English text first, fallback to transcript."""

    source_text = translated_text if translated_text.strip() else transcript_text
    normalized = _normalize_text(source_text)
    groups = config.speech_keyword_groups
    core_regex_hits = _find_core_regex_hits(normalized, config)

    help_hits = _find_hits(normalized, groups.get("help", ()))
    fall_hits = _find_hits(normalized, groups.get("fall", ())) + _find_hits(normalized, groups.get("fall_phrases", ()))
    cannot_breathe_hits = _find_hits(normalized, groups.get("cannot_breathe", ())) + _find_hits(
        normalized, groups.get("breathing_difficulty", ())
    )
    chest_pain_hits = _find_hits(normalized, groups.get("chest_pain", ()))
    generic_distress_hits = _find_hits(normalized, groups.get("distress_generic", ()))
    boost_hits = _find_hits(normalized, groups.get("distress_boost", ()))

    # Deduplicate each bucket while preserving order.
    help_hits = list(dict.fromkeys(help_hits))
    fall_hits = list(dict.fromkeys(fall_hits))
    cannot_breathe_hits = list(dict.fromkeys(cannot_breathe_hits))
    chest_pain_hits = list(dict.fromkeys(chest_pain_hits))
    generic_distress_hits = list(dict.fromkeys(generic_distress_hits))
    boost_hits = list(dict.fromkeys(boost_hits))

    # Keep insertion order while removing duplicates.
    keyword_hits = list(
        dict.fromkeys(
            help_hits
            + fall_hits
            + cannot_breathe_hits
            + chest_pain_hits
            + generic_distress_hits
            + boost_hits
            + core_regex_hits.get("help", [])
            + core_regex_hits.get("fall", [])
            + core_regex_hits.get("cannot_breathe", [])
        )
    )
    return {
        # "help" is treated as a general distress flag in this compact schema.
        "help_keyword_detected": len(help_hits) > 0
        or len(generic_distress_hits) > 0
        or len(boost_hits) > 0
        or len(core_regex_hits.get("help", [])) > 0,
        "fall_keyword_detected": len(fall_hits) > 0 or len(core_regex_hits.get("fall", [])) > 0,
        "cannot_breathe_keyword_detected": len(cannot_breathe_hits) > 0
        or len(chest_pain_hits) > 0
        or len(core_regex_hits.get("cannot_breathe", [])) > 0,
        "keyword_hits": keyword_hits,
    }
