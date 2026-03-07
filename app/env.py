"""Environment loading helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_dotenv_loaded() -> None:
    """Load `.env` once per process if python-dotenv is available."""

    try:
        from dotenv import find_dotenv, load_dotenv
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        pass

    if os.getenv("OPENAI_API_KEY"):
        return

    # Fallback parser when python-dotenv is missing or did not find a file.
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip("'\"")
            if key and not os.environ.get(key):
                os.environ[key] = value
    except Exception:
        return
