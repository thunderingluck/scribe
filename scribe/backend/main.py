"""
Scribe backend — FastAPI app.

Endpoints:
  POST /upload          Upload a JSONL essay dataset
  POST /load-demo       Load essays from the bundled demo file
  POST /preprocess      Convert raw essays to OpenAI fine-tuning format
  POST /fine-tune       Start an OpenAI fine-tuning job
  GET  /fine-tune/status  Poll fine-tuning job status
  POST /generate        Generate a new essay using the fine-tuned model
  GET  /state           Return current persisted state (for UI polling)
  POST /reset           Clear all state
"""

import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import openai
import state as st
import preprocessing
import retrieval

load_dotenv()

app = FastAPI(title="Scribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_DIR = Path(__file__).parent / "state"
RAW_ESSAYS_PATH = STATE_DIR / "raw_essays.jsonl"
TRAINING_DATA_PATH = STATE_DIR / "training_data.jsonl"

# Demo data lives three levels up from backend/: backend/ -> scribe/ -> scribe/ -> data/
DEMO_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "essays.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_openai_client():
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/state")
def get_state():
    return st.get()


@app.post("/reset")
def reset_state():
    for p in [RAW_ESSAYS_PATH, TRAINING_DATA_PATH]:
        if p.exists():
            p.unlink()
    return st.reset()


@app.post("/upload")
async def upload_essays(file: UploadFile = File(...)):
    """Accept a JSONL file upload and save it as the raw essays dataset."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    # Validate: each line must be valid JSON
    lines = [l for l in content.decode("utf-8").splitlines() if l.strip()]
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON on line {i + 1}: {e}")
    RAW_ESSAYS_PATH.write_bytes(content)
    st.update({"raw_essays_path": str(RAW_ESSAYS_PATH), "training_data_path": None,
                "ft_job_id": None, "ft_file_id": None, "model_id": None})
    return {"message": f"Uploaded {len(lines)} essays.", "count": len(lines)}


@app.post("/load-demo")
def load_demo():
    """Copy the bundled demo essays.jsonl into the state directory."""
    if not DEMO_PATH.exists():
        raise HTTPException(404, f"Demo file not found at {DEMO_PATH}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMO_PATH, RAW_ESSAYS_PATH)
    lines = [l for l in RAW_ESSAYS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    st.update({"raw_essays_path": str(RAW_ESSAYS_PATH), "training_data_path": None,
                "ft_job_id": None, "ft_file_id": None, "model_id": None})
    return {"message": f"Loaded {len(lines)} demo essays.", "count": len(lines)}


@app.post("/preprocess")
def preprocess():
    """Convert raw essays to OpenAI fine-tuning format."""
    cur = st.get()
    raw_path = cur.get("raw_essays_path")
    if not raw_path or not Path(raw_path).exists():
        raise HTTPException(400, "No essays loaded. Upload or load demo first.")
    count = preprocessing.preprocess_file(Path(raw_path), TRAINING_DATA_PATH)
    st.update({"training_data_path": str(TRAINING_DATA_PATH)})
    return {"message": f"Preprocessed {count} examples.", "count": count,
            "output_path": str(TRAINING_DATA_PATH)}


@app.post("/fine-tune")
def start_fine_tune():
    """Upload training data to OpenAI and start a fine-tuning job."""
    cur = st.get()
    training_path = cur.get("training_data_path")
    if not training_path or not Path(training_path).exists():
        raise HTTPException(400, "No training data. Run preprocess first.")

    client = _get_openai_client()

    # Upload training file to OpenAI
    with open(training_path, "rb") as f:
        upload_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = upload_resp.id

    # Create fine-tuning job
    # Architectural note: add hyperparameter control (n_epochs, learning_rate_multiplier)
    # here when experimenting with training configurations.
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=cur["base_model"],
    )

    st.update({"ft_job_id": job.id, "ft_file_id": file_id, "model_id": None})
    return {"message": "Fine-tuning job started.", "job_id": job.id, "status": job.status}


@app.get("/fine-tune/status")
def fine_tune_status():
    """Poll the OpenAI fine-tuning job for its current status."""
    cur = st.get()
    job_id = cur.get("ft_job_id")
    if not job_id:
        raise HTTPException(400, "No fine-tuning job found. Start one first.")

    client = _get_openai_client()
    job = client.fine_tuning.jobs.retrieve(job_id)

    # Persist the model ID once training completes
    if job.status == "succeeded" and job.fine_tuned_model:
        st.update({"model_id": job.fine_tuned_model})

    return {
        "job_id": job.id,
        "status": job.status,
        "model_id": job.fine_tuned_model,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }


class GenerateRequest(BaseModel):
    prompt: str
    top_k_context: int = 3  # how many excerpts to include as context


@app.post("/generate")
def generate_essay(req: GenerateRequest):
    """Generate a new essay using the fine-tuned model."""
    cur = st.get()
    model_id = cur.get("model_id")
    if not model_id:
        raise HTTPException(400, "No fine-tuned model available. Complete fine-tuning first.")

    # Retrieve relevant context excerpts (keyword heuristic)
    # Architectural note: swap retrieval.retrieve() for an embedding-based
    # implementation when adding a vector database.
    context_excerpts: list[str] = []
    raw_path = cur.get("raw_essays_path")
    if raw_path and Path(raw_path).exists() and req.top_k_context > 0:
        essays = retrieval.load_essays(Path(raw_path))
        context_excerpts = retrieval.retrieve(req.prompt, essays, top_k=req.top_k_context)

    # Build system prompt
    system_content = (
        f"You are embodying the user, writing their essay on \"{req.prompt}\". "
        "Use relevant facts and stylistic cues from their previous essays."
    )
    if context_excerpts:
        excerpts_text = "\n\n".join(f"- {e}" for e in context_excerpts)
        system_content += f"\n\nRelevant excerpts from previous essays:\n{excerpts_text}"

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Write an essay on: {req.prompt}"},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    essay = response.choices[0].message.content
    return {
        "essay": essay,
        "model_id": model_id,
        "context_excerpts": context_excerpts,
    }
