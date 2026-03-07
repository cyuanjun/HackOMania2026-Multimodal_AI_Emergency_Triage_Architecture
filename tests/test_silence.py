import numpy as np

from app.audio.silence import detect_silence_after_impact


def test_detect_silence_after_impact_flags_trailing_quiet_region() -> None:
    sample_rate = 16_000
    impact = np.ones(sample_rate // 30, dtype=np.float32) * 0.95
    silence = np.zeros(sample_rate * 2, dtype=np.float32)
    audio = np.concatenate([impact, silence])

    detected, confidence = detect_silence_after_impact(
        audio,
        sample_rate,
        impact_index=10,
        top_db=35.0,
        min_silence_sec=1.0,
    )

    assert detected is True
    assert confidence >= 0.8

