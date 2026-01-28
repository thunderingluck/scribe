# api.py
"""
Public HTTP API for Scribe inference (FastAPI + Uvicorn).

This file is intentionally self-contained (doesn't require refactoring infer.py).
It loads:
- a local base model directory (e.g., hf_models/mistral7b_instruct_v0_2)
- a local LoRA adapter directory (e.g., outputs/20260126_063833/adapter)

Endpoints:
- GET  /health
- POST /generate  -> { "text": "..." }

Run (on VM):
  source .venv/bin/activate
  pip install fastapi uvicorn pydantic transformers peft torch
  BASE_MODEL=hf_models/mistral7b_instruct_v0_2 \
  ADAPTER_PATH=outputs/20260126_063833/adapter \
  uvicorn api:app --host 127.0.0.1 --port 8000

Then put nginx in front on 443.

Optional security:
  export API_KEY="some-long-random-string"
  Then clients must send header:  X-API-Key: <API_KEY>
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------------------------
# Config (env-driven)
# -------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "hf_models/mistral7b_instruct_v0_2")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "outputs/20260126_063833/adapter")
API_KEY = os.environ.get("API_KEY")  # if set, require X-API-Key header
TRANSFORMERS_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "1")  # default offline

# Set offline mode by default to avoid surprise downloads
os.environ.setdefault("TRANSFORMERS_OFFLINE", TRANSFORMERS_OFFLINE)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Scribe API", version="0.1.0")

tokenizer = None
model = None
device = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(400, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    latency_s: float


def _require_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY is None:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _pick_dtype() -> torch.dtype:
    # Prefer bf16 when available (common on A100/H100), else fp16.
    if torch.cuda.is_available():
        # bf16 availability depends on GPU architecture + torch build
        if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere+ often OK
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_model_and_tokenizer() -> tuple[AutoTokenizer, torch.nn.Module, str]:
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)

    # Base model
    dtype = _pick_dtype()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )

    # Attach LoRA adapter
    peft = PeftModel.from_pretrained(
        base,
        ADAPTER_PATH,
        local_files_only=True,
    )

    # If we’re CPU-only, ensure model is on CPU
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        peft = peft.to("cpu")

    peft.eval()

    # Some tokenizers (Mistral/Llama style) prefer padding side left for generation in batches;
    # we're single-prompt most of the time, but this is safe.
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    return tok, peft, dev


@torch.inference_mode()
def _generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    assert tokenizer is not None and model is not None

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    do_sample = temperature > 0.0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated continuation (optional).
    # If you want the full prompt+completion, just decode outputs[0] directly.
    gen_tokens = outputs[0]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text


@app.on_event("startup")
def startup() -> None:
    global tokenizer, model, device
    tokenizer, model, device = _load_model_and_tokenizer()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "device": device,
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> GenerateResponse:
    _require_api_key(x_api_key)

    t0 = time.time()
    text = _generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    latency = time.time() - t0
    return GenerateResponse(text=text, latency_s=latency)
