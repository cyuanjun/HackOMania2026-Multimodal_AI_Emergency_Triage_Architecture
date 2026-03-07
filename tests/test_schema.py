from pathlib import Path
import math
import struct
import wave

import numpy as np

from app.pipeline import analyze_audio


def test_pipeline_returns_expected_schema(tmp_path: Path) -> None:
    sample_rate = 16_000
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = []
        for i in range(sample_rate):
            value = int(0.05 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.append(struct.pack("<h", value))
        wav_file.writeframes(b"".join(frames))

    result = analyze_audio(path)

    assert set(result.keys()) == {
        "audio_meta",
        "non_speech_events",
        "acoustic_features",
        "language",
        "transcript",
        "speech_features",
        "explanations",
        "model_info",
    }
    assert set(result["audio_meta"].keys()) == {
        "duration_sec",
        "sample_rate",
        "channels",
        "quality_ok",
        "quality_issues",
    }
    assert set(result["non_speech_events"].keys()) == {
        "crying_detected",
        "crying_confidence",
        "shouting_detected",
        "shouting_confidence",
        "impact_detected",
        "impact_confidence",
        "fall_sound_detected",
        "fall_sound_confidence",
        "breathing_irregularity_detected",
        "breathing_irregularity_confidence",
        "silence_after_impact_detected",
        "silence_after_impact_confidence",
    }
    assert set(result["acoustic_features"].keys()) == {
        "rms_energy_mean",
        "rms_energy_max",
        "silence_ratio",
        "peak_count",
        "sudden_impact_score",
        "breathing_variability_score",
    }
    assert set(result["language"].keys()) == {
        "detected_language",
        "detected_dialect",
        "confidence",
    }
    assert set(result["transcript"].keys()) == {
        "text",
        "translated_text",
        "asr_confidence",
    }
    assert set(result["speech_features"].keys()) == {
        "help_keyword_detected",
        "fall_keyword_detected",
        "cannot_breathe_keyword_detected",
        "keyword_hits",
        "voice_strength_score",
        "speech_urgency_score",
    }
    assert set(result["model_info"].keys()) == {"event_model", "explanations_source", "version"}
