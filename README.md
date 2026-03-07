# Personal Alert Button Non-Speech Audio Intelligence

This repository contains a production-style hackathon module for the Personal Alert Button (PAB) non-speech audio component. It analyzes an input audio clip and returns a JSON-serializable Python `dict` containing non-speech distress signals for downstream fusion.

The implementation is intentionally pragmatic:

- core event detection uses transparent heuristics that are easy to debug during a demo
- an optional YAMNet hook is attempted only if `tensorflow` and `tensorflow_hub` are already available
- failures inside one submodule never crash the whole pipeline

## Project layout

```text
app/
  config.py
  pipeline.py
  audio/
    loader.py
    quality.py
    features.py
    silence.py
    event_detection.py
demo.py
tests/
requirements.txt
```

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want optional YAMNet classification, install `tensorflow` and `tensorflow_hub` separately. The pipeline does not require them.

For `mp3` and `m4a` support, make sure your environment also has a working decoder backend. In practice:

```bash
brew install ffmpeg
```

`wav` works through the built-in fallback path. `mp3` and `m4a` typically rely on `soundfile` or `librosa` plus system codec support.

## Usage

```python
from app.pipeline import analyze_non_speech

result = analyze_non_speech("sample.wav")
print(result)
```

CLI demo:

```bash
python demo.py sample.wav
python demo.py sample.mp3
python demo.py sample.m4a
```

## Output schema

```python
{
  "audio_meta": {
    "duration_sec": float,
    "sample_rate": int,
    "channels": int,
    "quality_ok": bool,
    "quality_issues": list[str]
  },
  "non_speech_events": {
    "crying_detected": bool,
    "crying_confidence": float,
    "shouting_detected": bool,
    "shouting_confidence": float,
    "impact_detected": bool,
    "impact_confidence": float,
    "fall_sound_detected": bool,
    "fall_sound_confidence": float,
    "breathing_irregularity_detected": bool,
    "breathing_irregularity_confidence": float,
    "silence_after_impact_detected": bool,
    "silence_after_impact_confidence": float
  },
  "acoustic_features": {
    "rms_energy_mean": float,
    "rms_energy_max": float,
    "silence_ratio": float,
    "peak_count": int,
    "sudden_impact_score": float,
    "breathing_variability_score": float
  },
  "explanations": list[str],
  "model_info": {
    "event_model": str,
    "version": str
  }
}
```

## Heuristic logic

- `impact_detected`: weighted combination of sudden peaks, onset strength, and broadband burst energy
- `silence_after_impact_detected`: low-energy segment after the first strong impact candidate
- `fall_sound_detected`: weighted combination of impact confidence and post-impact silence confidence
- `shouting_detected`: sustained energy, peak energy, onset activity, and optional classifier hints
- `crying_detected`: unstable modulation, spectral spread, and optional classifier hints
- `breathing_irregularity_detected`: envelope variability with optional classifier hints

## Notes for extension

- `app/audio/event_detection.py` contains the swap point for a future trained classifier.
- `app/audio/features.py` is the main place to add stronger acoustic descriptors later.
- confidence values are clamped to `[0, 1]` and remain JSON-safe.

## Tests

```bash
pytest
```
