"""
Preprocessing module: converts raw essays into OpenAI fine-tuning format.

Expected input schema (one JSON object per line):
  {"prompt": "<essay prompt / topic>", "essay": "<full essay text>"}

Output format (OpenAI chat fine-tuning, one JSON object per line):
  {
    "messages": [
      {"role": "system", "content": "<system prompt>"},
      {"role": "user",   "content": "Write an essay on: <prompt>"},
      {"role": "assistant", "content": "<essay>"}
    ]
  }

Architectural note:
  The system prompt is kept minimal here. A future improvement is to
  include rich style descriptors extracted from the corpus, or to treat
  the system prompt itself as a fine-tuning target (instruction fine-tuning
  variant where the system message is also part of the training data).
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are embodying the user, writing their essay. "
    "Replicate their voice, tone, and stylistic patterns faithfully."
)


def convert_essay_to_training_example(record: dict) -> dict:
    """Convert a single {'prompt', 'essay'} record to OpenAI chat format."""
    prompt = record.get("prompt", "").strip()
    essay = record.get("essay", "").strip()
    if not prompt or not essay:
        raise ValueError(f"Record missing 'prompt' or 'essay': {record}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Write an essay on: {prompt}"},
            {"role": "assistant", "content": essay},
        ]
    }


def preprocess_file(input_path: Path, output_path: Path) -> int:
    """
    Read raw essays JSONL and write OpenAI fine-tuning JSONL.
    Returns the number of examples written.
    """
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(convert_essay_to_training_example(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return len(examples)
