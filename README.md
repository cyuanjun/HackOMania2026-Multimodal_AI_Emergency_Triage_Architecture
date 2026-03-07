# Personal Alert Button Audio Intelligence Module

This repository provides a practical, hackathon-ready audio pipeline for the Personal Alert Button (PAB) project.

The pipeline supports:
- non-speech distress event detection (impact, fall-like pattern, shouting-like pattern, crying-like pattern, breathing irregularity)
- transcript extraction
- language detection
- translation to English
- speech keyword detection (`help`, `fall`, `cannot breathe`)
- speech urgency scoring
- LLM-assisted operator-facing explanation generation with deterministic fallback

The output is always a JSON-serializable Python `dict`, and module failures are handled gracefully without crashing the full pipeline.

## Project Layout

```text
app/
  config.py
  env.py
  explanations.py
  pipeline.py
  audio/
    loader.py
    quality.py
    features.py
    silence.py
    event_detection.py
  speech/
    asr_openai.py
    keywords.py
    language.py
    translation.py
    urgency.py
demo.py
tests/
requirements.txt
```

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For `mp3` and `m4a`, install a system decoder:

```bash
brew install ffmpeg
```

## OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

The code automatically loads `.env` for ASR, translation, and LLM explanation generation.

## Usage

### Full Audio Pipeline (non-speech + transcript + translation)

```python
from app.pipeline import analyze_audio

result = analyze_audio("sample.wav")
print(result)
```

### Non-Speech Only

```python
from app.pipeline import analyze_non_speech

result = analyze_non_speech("sample.wav")
print(result)
```

### CLI Demo

```bash
python demo.py sample.wav
python demo.py sample.mp3
python demo.py sample.m4a
```

## Full Output Schema (`analyze_audio`)

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
  "language": {
    "detected_language": str,
    "detected_dialect": str,
    "confidence": float
  },
  "transcript": {
    "text": str,
    "translated_text": str,
    "asr_confidence": float | None
  },
  "speech_features": {
    "help_keyword_detected": bool,
    "fall_keyword_detected": bool,
    "cannot_breathe_keyword_detected": bool,
    "keyword_hits": list[str],
    "voice_strength_score": float,
    "speech_urgency_score": float
  },
  "explanations": list[str],
  "model_info": {
    "event_model": str,
    "version": str
  }
}
```

## Notes

- Non-speech detection is heuristic-first and demo-friendly.
- Optional YAMNet signals are used when TensorFlow + TensorFlow Hub are available.
- Transcript/translation uses OpenAI APIs when `OPENAI_API_KEY` is set.
- Explanations are generated from existing structured signals (non-speech + speech + fusion hints). The LLM is not used for raw event detection.
- If OpenAI explanation generation fails, the pipeline falls back to deterministic rule-based explanations.
- Keyword matching uses a grouped config structure in `app/config.py` (`speech_keyword_groups`, `speech_core_regex_patterns`) for easier maintenance.
- Matching strategy is heuristic-based: mostly simple matching, with conservative regex relaxation on core emergency phrases.
- If ASR/translation is unavailable, the pipeline still returns a valid output with safe defaults and warning messages in `audio_meta.quality_issues`.

## Tests

```bash
pytest
```

Keyword regression test:

```bash
pytest tests/test_keywords.py -q
```
