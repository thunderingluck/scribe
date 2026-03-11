#!/usr/bin/env python3
"""
infer.py — Run inference with a LoRA adapter for essay-style generation (Colab-matched).

Training/inference prompt format (matches the attached .ipynb):
  Prompt: <prompt>

  Essay:
  <model continues here>

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
import time


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# End-of-essay marker
END_MARKER = "<END_ESSAY>"

# ----------------------------
# Path helpers
# ----------------------------
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


# ----------------------------
# Prompt formatting (Colab-matched)
# ----------------------------
def build_prompt(user_prompt: str) -> str:
    # Exact formatting matters: match whatever you trained on in train.py / Colab
    return f"Prompt: {user_prompt.strip()}\n\nEssay:\n"


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return sys.stdin.read()


# ----------------------------
# Loading
# ----------------------------
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
    device_map: dict | str | None,
    compute_dtype: torch.dtype,
    use_8bit: bool,
    use_4bit: bool,
    local_only: bool,
) -> Any:
    common_kwargs: Dict[str, Any] = dict(
        use_safetensors=True,
        local_files_only=local_only,
        attn_implementation="flash_attention_2",
    )

    if device_map is not None:
        common_kwargs["device_map"] = device_map
        common_kwargs["low_cpu_mem_usage"] = True

    if use_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            dtype=compute_dtype,
            **common_kwargs,
        )

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            dtype=compute_dtype,
            **common_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=compute_dtype,
        **common_kwargs,
    )

# ----------------------------
# Generation
# ----------------------------
def run_once(
    *,
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    full_prompt = build_prompt(user_prompt)

    device = next(model.parameters()).device
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

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

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    t1 = time.perf_counter()
    in_len = inputs["input_ids"].shape[1]
    out_len = output_ids.shape[1]
    new_tokens = out_len - in_len
    print(f"[perf] prompt_tokens={in_len} new_tokens={new_tokens} time={t1-t0:.2f}s tok/s={new_tokens/(t1-t0+1e-9):.1f}",
      file=sys.stderr)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Colab-style: extract only the continuation after the prompt prefix
    if decoded.startswith(full_prompt):
        text = decoded[len(full_prompt):]
    else:
        # Fallback: if tokenization/decoding quirks remove or alter prefix,
        # do a best-effort split on the marker.
        marker = "\n\nEssay:\n"
        if marker in decoded:
            text = decoded.split(marker, 1)[1]
        else:
            text = decoded

    text = text.lstrip()

    # <<< ADD THIS: stop at END_MARKER >>>
    if END_MARKER in text:
        text = text.split(END_MARKER, 1)[0]

    return text.strip()


def main() -> None:
    # Add at start of main(), before model loading
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+ GPUs
    p = argparse.ArgumentParser()
    p.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes)")
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

    # Whether HF may download missing files
    local_only = not args.allow_download

    # Device/dtype decisions
    has_cuda = torch.cuda.is_available()

    if args.cpu or not has_cuda:
        device_map: str | None = None
        use_8bit = False
        use_4bit = False
        compute_dtype = torch.float32
    else:
        device_map = {"": 0}

        use_8bit = bool(args.load_8bit)
        use_4bit = (not args.no_4bit) and (not use_8bit)  # 8-bit overrides 4-bit

        compute_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float16    # Helpful diagnostics
    print(
        f"[info] cuda_available={has_cuda} cpu_forced={args.cpu} use_8bit={use_8bit} use_4bit={use_4bit}",
        file=sys.stderr,
    )
    if has_cuda:
        try:
            print(
                f"[info] gpu={torch.cuda.get_device_name(0)} bf16_supported={torch.cuda.is_bf16_supported()}",
                file=sys.stderr,
            )
        except Exception:
            pass
    print(
        f"[info] dtype={compute_dtype} device_map={device_map} local_files_only={local_only}",
        file=sys.stderr,
    )
    print(f"[info] base_model={args.base_model}", file=sys.stderr)
    print(f"[info] adapter_path={adapter_path}", file=sys.stderr)
    print(f"[info] tokenizer_path={tokenizer_path}", file=sys.stderr)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path, local_only=local_only)

    # Load base model
    base = load_base_model(
        args.base_model,
        device_map=device_map,
        compute_dtype=compute_dtype,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        local_only=local_only,
)
    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()

    # # After model.eval()
    # if hasattr(torch, 'compile') and not args.cpu:
    #     print("[info] Compiling model with torch.compile...", file=sys.stderr)
    #     model = torch.compile(model, mode="reduce-overhead")
    #     # Do a warmup inference to trigger compilation
    #     _ = model.generate(**tokenizer("warmup", return_tensors="pt").to(model.device), max_new_tokens=1)

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
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        )


if __name__ == "__main__":
    main()