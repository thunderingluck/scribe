import json
import random
from pathlib import Path

RANDOM_SEED = 42
TRAIN_FRACTION = 0.8  # matches the Colab 80/20 split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "essays.jsonl"
OUT_TRAIN = ROOT / "data" / "processed" / "train.jsonl"
OUT_VAL = ROOT / "data" / "processed" / "val.jsonl"

random.seed(RANDOM_SEED)


def load_prompt_essay_examples(path: Path):
    """
    Robust loader that matches the Colab notebook:
    - Treats any line starting with {"prompt": as the start of a new example.
    - If subsequent lines don't start with {"prompt":, they are treated as essay continuation lines.
    This supports "jsonl-ish" files where essays contain raw newlines.
    """
    examples = []
    current = None

    def is_prompt_line(line: str) -> bool:
        return line.lstrip().startswith('{"prompt":')

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            if is_prompt_line(line):
                # Save previous
                if current is not None:
                    examples.append(current)

                # Parse the new line as JSON (best effort)
                try:
                    obj = json.loads(line)
                except Exception:
                    # If it's malformed, fall back to empty fields
                    obj = {"prompt": "", "essay": ""}

                current = {
                    "prompt": str(obj.get("prompt", "")).strip(),
                    "essay": str(obj.get("essay", "")).strip(),
                }
            else:
                # Continuation of essay text
                if current is not None:
                    line_stripped = line.rstrip("\n")
                    if current["essay"]:
                        current["essay"] += "\n" + line_stripped
                    else:
                        current["essay"] = line_stripped

    if current is not None:
        examples.append(current)

    return examples


rows = load_prompt_essay_examples(SRC)
assert len(rows) >= 2, f"Not enough essays to split (found {len(rows)})"

random.shuffle(rows)

train_count = max(1, int(len(rows) * TRAIN_FRACTION))
train_rows = rows[:train_count]
val_rows = rows[train_count:]
assert len(val_rows) >= 1, "Split produced empty val set; reduce TRAIN_FRACTION or add more data."

OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

with OUT_TRAIN.open("w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with OUT_VAL.open("w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(train_rows)} train examples → {OUT_TRAIN}")
print(f"Wrote {len(val_rows)} val examples → {OUT_VAL}")