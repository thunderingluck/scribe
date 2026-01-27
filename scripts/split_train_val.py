import json
import random
from pathlib import Path

RANDOM_SEED = 42
TRAIN_COUNT = 10  # for 13 essays

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "essays.jsonl"
OUT_TRAIN = ROOT / "data" / "processed" / "train.jsonl"
OUT_VAL = ROOT / "data" / "processed" / "val.jsonl"

random.seed(RANDOM_SEED)

rows = []
with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

assert len(rows) >= TRAIN_COUNT + 1, "Not enough essays to split"

random.shuffle(rows)

train_rows = rows[:TRAIN_COUNT]
val_rows = rows[TRAIN_COUNT:]

with OUT_TRAIN.open("w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with OUT_VAL.open("w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(train_rows)} train examples → {OUT_TRAIN}")
print(f"Wrote {len(val_rows)} val examples → {OUT_VAL}")
