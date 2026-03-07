"""CLI demo for the non-speech audio intelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from app.pipeline import analyze_non_speech


def _resolve_audio_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.exists():
        return str(path)

    sample_audio_path = Path("sample_audio") / raw_path
    if sample_audio_path.exists():
        return str(sample_audio_path)

    return raw_path


def main() -> int:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    result = analyze_non_speech(_resolve_audio_path(audio_path))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
