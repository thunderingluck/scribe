# Scribe — Personalized Essay Generation

A minimal full-stack prototype that fine-tunes an OpenAI model on a user's essays and generates new ones in their style.

---

## Architecture

```
scribe/
├── backend/          # FastAPI Python server
│   ├── main.py       # Routes / app entry point
│   ├── preprocessing.py  # Raw → OpenAI fine-tuning format
│   ├── retrieval.py  # Keyword-based context retrieval
│   ├── state.py      # Disk-based state management
│   ├── state/        # Runtime state (state.json, training data)
│   └── requirements.txt
├── frontend/         # Vanilla HTML/JS, no framework
│   ├── index.html
│   └── app.js
├── .env.example
└── README.md
```

---

## Input Data Schema

Raw essay data is JSONL with one record per line:

```json
{"prompt": "Your essay topic or question", "essay": "The full essay text"}
```

The demo dataset lives at `../data/processed/essays.jsonl` relative to this directory.

---

## Preprocessing

`preprocessing.py` converts each `{"prompt", "essay"}` record into OpenAI chat fine-tuning format:

```json
{
  "messages": [
    {"role": "system", "content": "You are embodying the user, writing their essay. Replicate their voice, tone, and stylistic patterns faithfully."},
    {"role": "user",   "content": "Write an essay on: <prompt>"},
    {"role": "assistant", "content": "<essay>"}
  ]
}
```

This transformation is intentionally isolated in `preprocessing.py` so it can be modified independently — e.g. to include richer style descriptors or to experiment with instruction fine-tuning variants.

---

## Setup

### 1. Clone and enter the project

```bash
cd scribe   # this directory
```

### 2. Configure environment

```bash
cp .env.example backend/.env
# Edit backend/.env and set your OPENAI_API_KEY
```

### 3. Install backend dependencies

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Locally

### Start the backend

```bash
cd backend
uvicorn main:app --reload
# Runs on http://localhost:8000
```

### Open the frontend

Open `frontend/index.html` directly in your browser — no build step needed.

> If you encounter CORS issues, serve the frontend from a simple HTTP server:
> ```bash
> cd frontend
> python -m http.server 3000
> # Then open http://localhost:3000
> ```

---

## Usage Flow

| Step | Action | Backend endpoint |
|------|--------|-----------------|
| 1 | Upload a `.jsonl` file **or** click "Load demo dataset" | `POST /upload` or `POST /load-demo` |
| 2 | Click "Preprocess" | `POST /preprocess` |
| 3 | Click "Start fine-tuning" | `POST /fine-tune` |
| 4 | Click "Refresh status" until status is `succeeded` | `GET /fine-tune/status` |
| 5 | Enter a prompt and click "Generate" | `POST /generate` |

Fine-tuning jobs on `gpt-4o-mini` typically complete in 10–30 minutes.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/state` | Current persisted state |
| POST | `/reset` | Clear all state and local files |
| POST | `/upload` | Upload JSONL essay dataset |
| POST | `/load-demo` | Load bundled demo dataset |
| POST | `/preprocess` | Convert to fine-tuning format |
| POST | `/fine-tune` | Start OpenAI fine-tuning job |
| GET | `/fine-tune/status` | Poll job status |
| POST | `/generate` | Generate essay `{"prompt": "...", "top_k_context": 3}` |

---

## Retrieval (Context Injection)

At generation time, `retrieval.py` selects up to `top_k_context` (default 3) essays whose topics are most relevant to the new prompt, using word-intersection scoring. Short excerpts are injected into the system prompt as style/fact context.

This is intentionally simple — the module's interface is isolated so it can be replaced with embedding-based retrieval (FAISS, pgvector, etc.) without touching the API layer.

More details:
The flow in retrieval.py is:

  1. Load all raw {"prompt", "essay"} records from the file
  2. For each essay, compute a word-intersection (Jaccard-like) score between the new prompt's words and the combined words of that essay's prompt +   
  body
  3. Return the first ~300 characters of the top k essays by score

  The scoring function in retrieval.py:

  def _score(query_tokens: set[str], essay: dict) -> float:
      prompt_tokens = _tokenize(essay.get("prompt", ""))
      essay_tokens  = _tokenize(essay.get("essay", ""))
      combined = prompt_tokens | essay_tokens
      return len(query_tokens & combined) / len(query_tokens | combined)

  So if you generate an essay on "Why I want to work in AI", it scores every stored essay by how many words overlap with that phrase, then injects the 
  top 3 excerpts into the system prompt as context.

  Limitations of this approach:
  - It's purely lexical — no semantic understanding (e.g. "machine learning" won't match "AI")
  - Scores against the full essay body, not just the topic, so a long essay can dominate just by having many overlapping common words
  - Truncates at 300 chars, which may cut off mid-sentence

---

## Extension Points

- **Vector retrieval**: Replace `retrieval.retrieve()` with an embedding-based implementation.
- **Auth**: Add FastAPI middleware and replace `state.py` with a per-user store.
- **Database**: Swap `state.py` for SQLite or Postgres.
- **Preprocessing variants**: Add richer system prompts or instruction fine-tuning in `preprocessing.py`.
- **Hyperparameter control**: Expose `n_epochs` / `learning_rate_multiplier` on the `/fine-tune` endpoint.
