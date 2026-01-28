## Scribe

Fine-tune a LoRA adapter to imitate a user's essay-writing style and run inference locally.

### Requirements
- Python 3.10+
- GPU recommended for training/inference

Install deps:
```bash
pip install -r requirements.txt
```

### Data format
Training data is JSONL with one example per line:
```json
{"prompt": "Your essay prompt here", "essay": "The target essay text here"}
```

Default paths:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl` (optional)

### Train
```bash
python train.py --base_model mistralai/Mistral-7B-Instruct-v0.2
```

Outputs:
- `outputs/adapters/<run_id>/adapter/`
- `outputs/adapters/<run_id>/tokenizer/`
- `outputs/logs/<run_id>.log`

## Models

Download the base model locally (not committed to git):

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir hf_models/mistral7b_instruct_v0_2


### Inference
Basic (single prompt):
```bash
py infer.py --base_model mistralai/Mistral-7B-Instruct-v0.2 --adapter_path outputs\adapters\<run_id>\adapter --prompt "Write a cover letter for XYZ company"
```

Interactive:
```bash
py infer.py --base_model mistralai/Mistral-7B-Instruct-v0.2 --adapter_path outputs\adapters\<run_id>\adapter --interactive
```

Notes:
- The adapter must match the exact base model it was trained on.
- `infer.py` prefers safetensors and uses local cache only by default. Use `--allow_download` to fetch missing files.
- For CPU, use `--cpu` (slow for 7B models).
