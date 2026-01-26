#!/usr/bin/env python3
"""
train.py — LoRA/QLoRA fine-tuning for essay style imitation

Expects JSONL lines like:
{"prompt": "...", "essay": "..."}

Default paths:
  data/processed/train.jsonl
  data/processed/val.jsonl   (optional)

Outputs:
  outputs/adapters/<run_id>/
  outputs/logs/<run_id>.log

Run (from repo root, inside your GPU Slurm shell + venv):
  python train.py --base_model mistralai/Mistral-7B-Instruct-v0.2

Notes:
- Uses QLoRA (4-bit) by default to fit on a single 24GB GPU (A30).
- Trains LoRA adapters only; base model weights stay unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
            if "prompt" not in obj or "essay" not in obj:
                raise ValueError(f"Line {i} of {path} missing 'prompt' or 'essay' keys.")
            rows.append({"prompt": str(obj["prompt"]), "essay": str(obj["essay"])})
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def build_text_example(
    prompt: str,
    essay: str,
    tokenizer: Any,
    system_prompt: str,
) -> str:
    """
    Creates a single training text string.

    If tokenizer supports chat templates, use it. Otherwise fall back to a plain format.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": essay.strip()},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        # add_generation_prompt=False because we already include the assistant content (label)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Fallback formatting (works for any CausalLM tokenizer)
    return (
        f"### System:\n{system_prompt.strip()}\n\n"
        f"### User:\n{prompt.strip()}\n\n"
        f"### Assistant:\n{essay.strip()}\n"
    )


def pick_lora_target_modules(model_name: str) -> List[str]:
    """
    Best-effort defaults that work for common decoder-only LLMs.
    You can override with --lora_targets.
    """
    name = model_name.lower()
    # Mistral/Llama-like
    if "mistral" in name or "llama" in name or "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Generic fallback
    return ["q_proj", "v_proj"]


@dataclass
class TokenizedBatch:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]


def tokenize_dataset(ds: Dataset, tokenizer: Any, max_seq_len: int) -> Dataset:
    def _tok(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_attention_mask=True,
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }

    return ds.map(_tok, batched=True, remove_columns=ds.column_names)



# ----------------------------
# Main
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="HF model name or local path")
    p.add_argument("--train_path", type=str, default="data/processed/train.jsonl")
    p.add_argument("--val_path", type=str, default="data/processed/val.jsonl")
    p.add_argument("--output_dir", type=str, default="outputs/adapters")
    p.add_argument("--log_dir", type=str, default="outputs/logs")

    p.add_argument("--run_id", type=str, default="", help="Optional run id; default uses timestamp")
    p.add_argument("--seed", type=int, default=42)

    # Data / tokenization
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--system_prompt", type=str, default=(
        "You are an essay-writing assistant. Write high-quality essays in the user's style. "
        "Match the user's tone, rhythm, and rhetorical habits. "
        "Do not copy exact sentences from the training essays; emulate style instead."
    ))

    # LoRA config
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="", help="Comma-separated module names to target")

    # Training config
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    # QLoRA / precision
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA (uses full precision)")
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported (else fp16)")

    args = p.parse_args()

    # Basic checks
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

    # Logging to file + stdout
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        # For many decoder-only models; safe for CausalLM fine-tune
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset rows
    train_rows = read_jsonl(train_path)
    val_rows: Optional[List[Dict[str, Any]]] = None
    if val_path.exists() and val_path.stat().st_size > 0:
        try:
            val_rows = read_jsonl(val_path)
        except Exception as e:
            log(f"WARNING: could not read val file ({e}); continuing without eval.")
            val_rows = None

    # Build "text" field
    def rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
        texts = [
            build_text_example(r["prompt"], r["essay"], tokenizer, args.system_prompt)
            for r in rows
        ]
        return Dataset.from_dict({"text": texts})

    train_ds = rows_to_dataset(train_rows)
    eval_ds = rows_to_dataset(val_rows) if val_rows else None

    log(f"Examples: train={len(train_ds)} eval={(len(eval_ds) if eval_ds else 0)}")

    # Tokenize
    train_ds = tokenize_dataset(train_ds, tokenizer, args.max_seq_len)
    if eval_ds:
        eval_ds = tokenize_dataset(eval_ds, tokenizer, args.max_seq_len)

    # Model load: QLoRA by default
    use_4bit = not args.no_4bit
    compute_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if use_4bit:
        log("Loading model in 4-bit (QLoRA).")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        log("Loading model in full precision (no 4-bit).")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

    # LoRA
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()] or pick_lora_target_modules(args.base_model)
    log(f"LoRA target modules: {targets}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Data collator for CausalLM (labels = input_ids)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"

    # Training args
    training_args = TrainingArguments(
        output_dir=str(out_root),
        logging_dir=str(log_root),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        **{eval_key: ("steps" if eval_ds else "no")},
        eval_steps=args.eval_steps if eval_ds else None,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        report_to=[],  # no wandb by default
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    log("Starting training...")
    trainer.train()

    log("Saving adapter + tokenizer...")
    # Save LoRA adapter weights (PEFT) and tokenizer
    model.save_pretrained(str(out_root / "adapter"))
    tokenizer.save_pretrained(str(out_root / "tokenizer"))

    # Write a tiny manifest
    manifest = {
        "run_id": run_id,
        "base_model": args.base_model,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path.exists() else None,
        "max_seq_len": args.max_seq_len,
        "system_prompt": args.system_prompt,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": targets,
            "qlora_4bit": use_4bit,
        },
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
        },
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"Done. Adapter saved to: {out_root / 'adapter'}")
    log(f"Tokenizer saved to: {out_root / 'tokenizer'}")


if __name__ == "__main__":
    main()
