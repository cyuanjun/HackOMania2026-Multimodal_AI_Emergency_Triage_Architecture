from app.config import DEFAULT_CONFIG
from app.speech.keywords import extract_speech_keywords


def test_keyword_regression_cases() -> None:
    cases = [
        ("Please help me now", (True, False, False)),
        ("Can you call an ambulance?", (True, False, False)),
        ("I have just fallen on the floor and can't get up", (True, True, False)),
        ("I slipped in the bathroom", (False, True, False)),
        ("I cant breathe", (False, False, True)),
        ("Breathing is very hard right now", (False, False, True)),
        ("There is shortness of breath", (False, False, True)),
        ("I have chest pain", (True, False, True)),
        ("I love helping people at the center", (False, False, False)),
        ("No emergency here, everything is fine", (True, False, False)),
    ]

    for text, expected in cases:
        result = extract_speech_keywords("", text, config=DEFAULT_CONFIG)
        got = (
            result["help_keyword_detected"],
            result["fall_keyword_detected"],
            result["cannot_breathe_keyword_detected"],
        )
        assert got == expected, f"text={text!r}, got={got}, expected={expected}, hits={result['keyword_hits']}"


def test_keywords_prefers_translation_over_raw_transcript() -> None:
    result = extract_speech_keywords(
        transcript_text="random transcript without emergency clues",
        translated_text="Please call an ambulance now",
        config=DEFAULT_CONFIG,
    )
    assert result["help_keyword_detected"] is True
    assert "ambulance" in result["keyword_hits"]


def test_keywords_fallbacks_to_transcript_when_translation_empty() -> None:
    result = extract_speech_keywords(
        transcript_text="I slipped and cannot get up",
        translated_text="",
        config=DEFAULT_CONFIG,
    )
    assert result["fall_keyword_detected"] is True

