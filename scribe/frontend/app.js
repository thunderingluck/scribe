/**
 * Scribe frontend — minimal vanilla JS, no framework.
 *
 * Talks to the FastAPI backend at API_BASE.
 * Architectural note: swap API_BASE for a deployed URL when hosting remotely.
 */

const API_BASE = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function setStatus(el, type, msg) {
  el.className = `status ${type}`;
  el.textContent = msg;
  el.style.display = "block";
}

function clearStatus(el) {
  el.className = "status";
  el.style.display = "none";
  el.textContent = "";
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Step 1: Load essays
// ---------------------------------------------------------------------------

document.getElementById("upload-btn").addEventListener("click", async () => {
  const input = document.getElementById("file-input");
  const statusEl = document.getElementById("load-status");
  if (!input.files.length) {
    setStatus(statusEl, "warn", "Please select a .jsonl file first.");
    return;
  }
  setStatus(statusEl, "info", "Uploading…");
  const formData = new FormData();
  formData.append("file", input.files[0]);
  try {
    const data = await apiFetch("/upload", { method: "POST", body: formData });
    setStatus(statusEl, "ok", `✓ ${data.message}`);
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
});

document.getElementById("demo-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("load-status");
  setStatus(statusEl, "info", "Loading demo dataset…");
  try {
    const data = await apiFetch("/load-demo", { method: "POST" });
    setStatus(statusEl, "ok", `✓ ${data.message}`);
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
});

// ---------------------------------------------------------------------------
// Step 2: Preprocess
// ---------------------------------------------------------------------------

document.getElementById("preprocess-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("preprocess-status");
  setStatus(statusEl, "info", "Preprocessing…");
  try {
    const data = await apiFetch("/preprocess", { method: "POST" });
    setStatus(statusEl, "ok", `✓ ${data.message}`);
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
});

// ---------------------------------------------------------------------------
// Step 3: Fine-tune
// ---------------------------------------------------------------------------

const pollBtn = document.getElementById("poll-btn");

document.getElementById("finetune-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("finetune-status");
  setStatus(statusEl, "info", "Starting fine-tuning job…");
  try {
    const data = await apiFetch("/fine-tune", { method: "POST" });
    setStatus(statusEl, "ok", `✓ Job started: ${data.job_id} — status: ${data.status}`);
    pollBtn.style.display = "inline-block";
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
});

function statusBadge(status) {
  const cls = {
    pending: "badge-pending",
    queued: "badge-pending",
    validating_files: "badge-running",
    running: "badge-running",
    succeeded: "badge-succeeded",
    failed: "badge-failed",
    cancelled: "badge-failed",
  }[status] || "badge-pending";
  return `<span class="badge ${cls}">${status}</span>`;
}

async function pollStatus() {
  const statusEl = document.getElementById("finetune-status");
  try {
    const data = await apiFetch("/fine-tune/status");
    const modelPart = data.model_id ? ` | model: <code>${data.model_id}</code>` : "";
    statusEl.innerHTML =
      `Job: <code>${data.job_id}</code>${statusBadge(data.status)}${modelPart}`;
    statusEl.className = "status info";
    statusEl.style.display = "block";
    if (data.status === "succeeded") {
      statusEl.className = "status ok";
    } else if (data.status === "failed" || data.status === "cancelled") {
      statusEl.className = "status error";
    }
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
}

pollBtn.addEventListener("click", pollStatus);

// ---------------------------------------------------------------------------
// Step 4: Generate
// ---------------------------------------------------------------------------

document.getElementById("generate-btn").addEventListener("click", async () => {
  const prompt = document.getElementById("prompt-input").value.trim();
  const statusEl = document.getElementById("generate-status");
  const outputEl = document.getElementById("essay-output");
  const contextDetails = document.getElementById("context-details");
  const contextPre = document.getElementById("context-pre");

  if (!prompt) {
    setStatus(statusEl, "warn", "Please enter an essay prompt.");
    return;
  }
  setStatus(statusEl, "info", "Generating essay…");
  outputEl.style.display = "none";
  contextDetails.style.display = "none";

  try {
    const data = await apiFetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, top_k_context: 3 }),
    });
    clearStatus(statusEl);
    outputEl.textContent = data.essay;
    outputEl.style.display = "block";

    if (data.context_excerpts && data.context_excerpts.length) {
      contextPre.textContent = data.context_excerpts.join("\n\n---\n\n");
      contextDetails.style.display = "block";
    }
  } catch (e) {
    setStatus(statusEl, "error", `Error: ${e.message}`);
  }
});

// ---------------------------------------------------------------------------
// On load: restore UI from persisted state
// ---------------------------------------------------------------------------

(async () => {
  try {
    const s = await apiFetch("/state");
    if (s.ft_job_id) {
      document.getElementById("poll-btn").style.display = "inline-block";
      await pollStatus();
    }
  } catch (_) {
    // Backend not running yet — silently ignore
  }
})();
