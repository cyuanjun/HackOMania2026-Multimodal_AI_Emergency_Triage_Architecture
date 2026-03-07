import numpy as np

from app.audio.event_detection import detect_events


def test_impact_detection_heuristic_detects_burst() -> None:
    sample_rate = 16_000
    quiet = np.zeros(sample_rate, dtype=np.float32)
    impact = np.concatenate(
        [
            np.zeros(3000, dtype=np.float32),
            np.ones(120, dtype=np.float32) * 0.98,
            np.zeros(sample_rate - 3120, dtype=np.float32),
        ]
    )
    audio = quiet + impact

    result = detect_events(audio, sample_rate)

    assert result["non_speech_events"]["impact_detected"] is True
    assert result["non_speech_events"]["impact_confidence"] >= 0.6
    assert result["acoustic_features"]["sudden_impact_score"] >= 0.45
