"""
Disk-based state management.

Persists fine-tuning job metadata to a local JSON file.
No database required.

Architectural note:
  Replace this module with a proper database (SQLite, Postgres) when
  adding multi-user support or auth.
"""

import json
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state" / "state.json"

_DEFAULTS = {
    "ft_job_id": None,       # OpenAI fine-tuning job ID
    "ft_file_id": None,      # OpenAI uploaded training file ID
    "model_id": None,        # Resulting fine-tuned model ID (set when job completes)
    "base_model": "gpt-4o-mini-2024-07-18",  # Base model used for fine-tuning
    "raw_essays_path": None,    # Path to uploaded/loaded raw essays
    "training_data_path": None, # Path to preprocessed training JSONL
}


def _load() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        # Merge with defaults so new keys are always present
        return {**_DEFAULTS, **data}
    return dict(_DEFAULTS)


def _save(data: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get() -> dict:
    return _load()


def update(patch: dict) -> dict:
    data = _load()
    data.update(patch)
    _save(data)
    return data


def reset() -> dict:
    _save(dict(_DEFAULTS))
    return dict(_DEFAULTS)
