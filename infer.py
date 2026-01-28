#!/usr/bin/env python3
"""
infer.py — Run inference with a LoRA adapter for essay-style generation.

Usage:
  python infer.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_path outputs/adapters/<run_id>/adapter \
    --prompt "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_adapter_path(path: Path) -> Path:
    """Accept either an adapter dir, or a run root that contains ./adapter/."""
    if (path / "adapter_config.json").exists():
        return path
    if (path / "adapter").is_dir() and (path / "adapter" / "adapter_config.json").exists():
        return path / "adapter"
    raise FileNotFoundError(f"Could not find adapter_config.json under: {path}")


def resolve_tokenizer_path(run_root: Path, base_model: str) -> str:
    """Prefer a saved tokenizer under run_root/tokenizer if present, else use base_model."""
    if (run_root / "tokenizer").is_dir():
        return str(run_root / "tokenizer")
    return base_model


def build_prompt(prompt: str, tokenizer: Any, system_prompt: str) -> str:
    """Build a chat prompt using the tokenizer's chat template if available."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback: a simple instruction format (may be suboptimal for Mistral-Instruct,
    # but keeps the script robust if chat templates are missing).
    return (
        f"### System:\n{system_prompt.strip()}\n\n"
        f"### User:\n{prompt.strip()}\n\n"
        f"### Assistant:\n"
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return sys.stdin.read()


def pick_compute_dtype(args: argparse.Namespace) -> torch.dtype:
    """Choose a reasonable dtype. On CPU, use fp32. On CUDA, prefer bf16 when available."""
    if args.cpu or not torch.cuda.is_available():
        return torch.float32
    if args.bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(tokenizer_path: str, local_only: bool) -> Any:
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=local_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    base_model: str,
    *,
    device_map: str,
    compute_dtype: torch.dtype,
    use_4bit: bool,
    local_only: bool,
) -> Any:
    # CPU: avoid unnecessary overhead and reduce peak RAM during load
    common_kwargs = dict(
        device_map=device_map,
        use_safetensors=True,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )

    if use_4bit:
        # 4-bit is CUDA-only in practice; keep it off for CPU
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            torch_dtype=compute_dtype,
            **common_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=compute_dtype,
        **common_kwargs,
    )


def run_once(
    *,
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    full_prompt = build_prompt(user_prompt, tokenizer, system_prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if top_k > 0:
        gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # If using chat templates, decoding often returns the full transcript.
    # For non-chat fallback format, strip the prompt prefix to return just assistant output.
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return text.strip()
    return text[len(full_prompt) :].lstrip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="HF model name or local path")
    p.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Adapter dir or run root with /adapter inside",
    )
    p.add_argument("--prompt", type=str, default="", help="User prompt (or provide via --prompt_file / stdin)")
    p.add_argument("--prompt_file", type=str, default="")
    p.add_argument("--interactive", action="store_true", help="Prompt in a loop for multiple requests")

    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are an essay-writing assistant. Write high-quality essays in the user's style. "
            "Match the user's tone, rhythm, and rhetorical habits. "
            "Do not copy exact sentences from training essays; emulate style instead."
        ),
    )

    p.add_argument("--max_new_tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading (GPU only)")
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported (else fp16) on CUDA")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference (disables 4-bit)")
    p.add_argument("--allow_download", action="store_true", help="Allow downloading model files if not cached")

    args = p.parse_args()

    # Read initial prompt (unless interactive)
    if args.interactive:
        prompt = ""
    else:
        prompt = read_prompt(args).strip()
        if not prompt:
            raise ValueError("Prompt is empty. Provide --prompt, --prompt_file, or stdin.")

    # Resolve paths
    adapter_root = Path(args.adapter_path)
    adapter_path = resolve_adapter_path(adapter_root)
    tokenizer_path = resolve_tokenizer_path(adapter_root, args.base_model)

    # Determinism
    torch.manual_seed(args.seed)

    local_only = not args.allow_download

    # Device/dtype decisions
    if args.cpu:
        device_map = "cpu"
        use_4bit = False
    else:
        device_map = "auto"
        use_4bit = (not args.no_4bit) and torch.cuda.is_available()

    compute_dtype = pick_compute_dtype(args)

    # Helpful diagnostics
    print(f"[info] cuda_available={torch.cuda.is_available()} cpu_forced={args.cpu} use_4bit={use_4bit}", file=sys.stderr)
    if torch.cuda.is_available():
        try:
            print(f"[info] gpu={torch.cuda.get_device_name(0)} bf16_supported={torch.cuda.is_bf16_supported()}", file=sys.stderr)
        except Exception:
            pass
    print(f"[info] dtype={compute_dtype} device_map={device_map} local_files_only={local_only}", file=sys.stderr)
    print(f"[info] base_model={args.base_model}", file=sys.stderr)
    print(f"[info] adapter_path={adapter_path}", file=sys.stderr)
    print(f"[info] tokenizer_path={tokenizer_path}", file=sys.stderr)

    # Load tokenizer first (fast; helps fail early if cache/download issues)
    tokenizer = load_tokenizer(tokenizer_path, local_only=local_only)

    # Load base model
    base = load_base_model(
        args.base_model,
        device_map=device_map,
        compute_dtype=compute_dtype,
        use_4bit=use_4bit,
        local_only=local_only,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()

    if args.interactive:
        print("Interactive mode. Type a prompt and press Enter. Ctrl+C to exit.")
        try:
            while True:
                user_prompt = input("> ").strip()
                if not user_prompt:
                    continue
                print(
                    run_once(
                        model=model,
                        tokenizer=tokenizer,
                        user_prompt=user_prompt,
                        system_prompt=args.system_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                    )
                )
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        print(
            run_once(
                model=model,
                tokenizer=tokenizer,
                user_prompt=prompt,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        )


if __name__ == "__main__":
    main()
