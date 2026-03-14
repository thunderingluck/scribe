#!/usr/bin/env python3
"""
train.py — LoRA/QLoRA fine-tuning for essay style imitation (Colab-matched)

Expects examples with:
{"prompt": "...", "essay": "..."}

Training text format (exactly like the attached Colab notebook):
  Prompt: <prompt>

  Essay:
  <essay>

Important: No system prompt. No chat template.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ----------------------------
# Utilities
# ----------------------------
def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_prompt_essay_examples(path: Path) -> List[Dict[str, str]]:
    """
    Robust loader matching the Colab notebook:
    - Any line starting with {"prompt": begins a new example
    - Other lines are appended to the current essay as raw continuation lines
    """
    examples: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None

    def is_prompt_line(line: str) -> bool:
        return line.lstrip().startswith('{"prompt":')

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            if is_prompt_line(line):
                if current is not None:
                    examples.append(current)

                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"prompt": "", "essay": ""}

                current = {
                    "prompt": str(obj.get("prompt", "")).strip(),
                    "essay": str(obj.get("essay", "")).strip(),
                }
            else:
                if current is not None:
                    line_stripped = line.rstrip("\n")
                    if current["essay"]:
                        current["essay"] += "\n" + line_stripped
                    else:
                        current["essay"] = line_stripped

    if current is not None:
        examples.append(current)

    # Basic validation
    cleaned = []
    for ex in examples:
        p, e = ex.get("prompt", "").strip(), ex.get("essay", "").strip()
        if p and e:
            cleaned.append({"prompt": p, "essay": e})

    if not cleaned:
        raise ValueError(f"No valid examples found in {path}")
    return cleaned


END_MARKER = "<END_ESSAY>"

def format_text(prompt: str, essay: str) -> str:
    return f"Prompt: {prompt.strip()}\n\nEssay:\n{essay.strip()}\n{END_MARKER}\n"


def tokenize_example(text: str, tokenizer: Any, max_length: int) -> Dict[str, Any]:
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # matches Colab behavior
    return tokens


def make_pad_collator(tokenizer):
    # Matches Colab: pad input_ids/attention_mask with pad_id; pad labels with -100
    pad_id = tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {}

        def pad_list(x, pad_val):
            return x + [pad_val] * (max_len - len(x))

        batch["input_ids"] = torch.tensor([pad_list(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([pad_list(f["attention_mask"], 0) for f in features], dtype=torch.long)
        else:
            # Create attention mask if missing
            batch["attention_mask"] = torch.tensor(
                [pad_list([1] * len(f["input_ids"]), 0) for f in features],
                dtype=torch.long,
            )

        batch["labels"] = torch.tensor([pad_list(f["labels"], -100) for f in features], dtype=torch.long)
        return batch

    return collate


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--base_model", type=str, required=True, help="HF model name or local path")
    p.add_argument("--train_path", type=str, default="../data/processed/train.jsonl")
    p.add_argument("--val_path", type=str, default="../data/processed/val.jsonl")

    p.add_argument("--output_dir", type=str, default="outputs/adapters")
    p.add_argument("--log_dir", type=str, default="outputs/logs")
    p.add_argument("--run_id", type=str, default="", help="Optional run id; default uses timestamp")
    p.add_argument("--seed", type=int, default=42)

    # Tokenization
    p.add_argument("--max_seq_len", type=int, default=2048)

    # LoRA (Colab defaults)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="q_proj,v_proj")  # matches Colab

    # Training (Colab-ish defaults)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)   # matches Colab
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging_steps", type=int, default=5)

    # QLoRA / precision
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA (uses full precision)")
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported (else fp16)")

    args = p.parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    run_id = args.run_id.strip() or now_run_id()
    out_root = Path(args.output_dir) / run_id
    log_root = Path(args.log_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{run_id}.log"

    def log(msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Run id: {run_id}")
    log(f"Base model: {args.base_model}")
    log(f"Train path: {train_path}")
    log(f"Val path:   {val_path} (exists={val_path.exists()})")
    log(f"Output dir: {out_root}")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data (robust parser)
    train_rows = load_prompt_essay_examples(train_path)
    val_rows: Optional[List[Dict[str, str]]] = None
    if val_path.exists() and val_path.stat().st_size > 0:
        try:
            val_rows = load_prompt_essay_examples(val_path)
        except Exception as e:
            log(f"WARNING: could not read val file ({e}); continuing without eval.")
            val_rows = None

    # Build HF datasets
    train_texts = [format_text(r["prompt"], r["essay"]) for r in train_rows]
    train_ds = Dataset.from_dict({"text": train_texts})

    eval_ds = None
    if val_rows:
        eval_texts = [format_text(r["prompt"], r["essay"]) for r in val_rows]
        eval_ds = Dataset.from_dict({"text": eval_texts})

    log(f"Examples: train={len(train_ds)} eval={(len(eval_ds) if eval_ds else 0)}")

    # Tokenize
    def _tok(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for text in batch["text"]:
            toks = tokenize_example(text, tokenizer, args.max_seq_len)
            out["input_ids"].append(toks["input_ids"])
            out["attention_mask"].append(toks.get("attention_mask", [1] * len(toks["input_ids"])))
            out["labels"].append(toks["labels"])
        return out

    train_ds = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(_tok, batched=True, remove_columns=eval_ds.column_names)

    # Model: QLoRA by default (Colab-style)
    use_4bit = not args.no_4bit
    compute_dtype = (
        torch.bfloat16
        if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    if use_4bit:
        log("Loading model in 4-bit (QLoRA).")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=True,
            device_map="auto",
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model)
    else:
        log("Loading model in full precision (no 4-bit).")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

    # LoRA config
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    log(f"LoRA target modules: {targets}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = make_pad_collator(tokenizer)

    # Training args (Colab-ish)
    training_args = TrainingArguments(
        output_dir=str(out_root),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    log("Starting training...")
    trainer.train()

    log("Saving adapter + tokenizer...")
    model.save_pretrained(str(out_root / "adapter"))
    tokenizer.save_pretrained(str(out_root / "tokenizer"))

    log(f"Done. Adapter saved to: {out_root / 'adapter'}")
    log(f"Tokenizer saved to: {out_root / 'tokenizer'}")


if __name__ == "__main__":
    main()