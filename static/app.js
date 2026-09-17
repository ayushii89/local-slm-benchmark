const MODEL_COLOR = {
  "llama3.2-3b": "var(--c-llama)",
  "phi4-mini": "var(--c-phi)",
  "mistral-7b": "var(--c-mistral)",
};

function getSessionId() {
  let id = localStorage.getItem("slm_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("slm_session_id", id);
  }
  return id;
}

function confidenceClass(confidence) {
  if (confidence == null) return "confidence-low";
  if (confidence >= 0.75) return "confidence-high";
  if (confidence >= 0.4) return "confidence-mid";
  return "confidence-low";
}

// ============ tabs ============
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");

      if (btn.dataset.tab === "dashboard" && !dashboardLoaded) {
        loadDashboard();
      }
    });
  });
}

// ============ chat ============
async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  const select = document.getElementById("modelSelect");
  select.innerHTML = "";
  for (const model of data.models) {
    const opt = document.createElement("option");
    opt.value = model;
    opt.textContent = model;
    if (model === data.default) opt.selected = true;
    select.appendChild(opt);
  }
}

function appendMessage(role, text) {
  const thread = document.getElementById("chatThread");
  const empty = thread.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  thread.appendChild(msg);
  thread.scrollTop = thread.scrollHeight;
  return msg;
}

function renderAssistantResult(msgEl, result) {
  if (!result.success) {
    msgEl.classList.add("failed");
    msgEl.querySelector(".msg-bubble").textContent = `Couldn't answer: ${result.error}`;
    return;
  }

  msgEl.querySelector(".msg-bubble").textContent = result.answer;

  const meta = document.createElement("div");
  meta.className = "msg-meta";

  if (result.rewritten_query) {
    const note = document.createElement("span");
    note.className = "rewrite-note";
    note.textContent = `rewritten: "${result.rewritten_query}"`;
    meta.appendChild(note);
  }

  const conf = document.createElement("span");
  conf.className = `confidence-pill ${confidenceClass(result.confidence)}`;
  conf.textContent = `confidence ${result.confidence.toFixed(2)}`;
  meta.appendChild(conf);

  for (const citation of result.citations) {
    const pill = document.createElement("span");
    pill.className = "citation-pill";
    pill.textContent = citation;
    meta.appendChild(pill);
  }

  if (result.citations.length === 0) {
    const pill = document.createElement("span");
    pill.className = "citation-pill";
    pill.textContent = "no citations";
    meta.appendChild(pill);
  }

  const timing = document.createElement("span");
  timing.className = "mono";
  timing.textContent = `${Math.round(result.retrieval_ms)}ms retrieval, ${Math.round(result.generation_ms)}ms gen`;
  meta.appendChild(timing);

  msgEl.appendChild(meta);
}

async function sendChat(question) {
  const model = document.getElementById("modelSelect").value;
  const sendBtn = document.querySelector("#chatForm button[type=submit]");
  sendBtn.disabled = true;

  appendMessage("user", question);
  const assistantMsg = appendMessage("assistant", "Thinking...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: getSessionId(), question, model }),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    renderAssistantResult(assistantMsg, result);
  } catch (err) {
    assistantMsg.classList.add("failed");
    assistantMsg.querySelector(".msg-bubble").textContent = `Request failed: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    document.getElementById("chatThread").scrollTop = document.getElementById("chatThread").scrollHeight;
  }
}

function initChat() {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    sendChat(question);
  });

  document.getElementById("resetBtn").addEventListener("click", async () => {
    await fetch(`/api/chat/reset?session_id=${encodeURIComponent(getSessionId())}`, { method: "POST" });
    document.getElementById("chatThread").innerHTML =
      '<div class="empty-state">Conversation reset. Ask a new question.</div>';
  });
}

// ============ dashboard ============
let dashboardLoaded = false;

function metricRow(label, metrics) {
  if (!metrics) return `<div class="stat"><span class="stat-value dim">&ndash;</span><span class="stat-label">${label}</span></div>`;
  return `<div class="stat"><span class="stat-value tabular">${metrics.mean.toFixed(1)}</span><span class="stat-label">${label}</span></div>`;
}

function renderPhase1(data) {
  if (!data.available) {
    return `<div class="section"><div class="section-title">Phase 1: Speed</div><div class="unavailable">No Phase 1 results yet -- run \`python benchmark.py\`.</div></div>`;
  }
  const cards = Object.entries(data.per_model).map(([model, m]) => `
    <div class="card" style="--card-color: ${MODEL_COLOR[model] || "var(--accent)"};">
      <div class="card-head"><span class="card-swatch"></span><span class="card-name">${model}</span></div>
      <div class="card-stats">
        ${metricRow("tok/s", m.metrics.tokens_per_sec)}
        ${metricRow("TTFT ms", m.metrics.ttft_ms)}
        ${metricRow("latency ms", m.metrics.total_latency_ms)}
      </div>
    </div>`).join("");
  return `<div class="section">
    <div class="section-title">Phase 1: Speed</div>
    <div class="section-tag">${data.source_file}</div>
    <div class="cards">${cards}</div>
  </div>`;
}

function renderPhase2(data) {
  if (!data.available) {
    return `<div class="section"><div class="section-title">Phase 2: Reliability</div><div class="unavailable">No Phase 2 results yet -- run \`python run_structured_eval.py\`.</div></div>`;
  }
  const cards = Object.entries(data.per_model).map(([model, m]) => `
    <div class="card" style="--card-color: ${MODEL_COLOR[model] || "var(--accent)"};">
      <div class="card-head"><span class="card-swatch"></span><span class="card-name">${model}</span></div>
      <div class="card-stats">
        <div class="stat"><span class="stat-value tabular">${m.first_try}/${m.total}</span><span class="stat-label">1st try</span></div>
        <div class="stat"><span class="stat-value tabular">${m.after_retry}/${m.total}</span><span class="stat-label">after retry</span></div>
        <div class="stat"><span class="stat-value tabular">${m.failed}/${m.total}</span><span class="stat-label">failed</span></div>
      </div>
    </div>`).join("");
  return `<div class="section">
    <div class="section-title">Phase 2: Reliability</div>
    <div class="section-tag">${data.source_file}</div>
    <div class="cards">${cards}</div>
  </div>`;
}

function renderPhase3(data) {
  if (!data.available) {
    return `<div class="section"><div class="section-title">Phase 3: Comparison</div><div class="unavailable">No Phase 3 results yet -- run \`python run_comparison.py\`.</div></div>`;
  }
  const cards = Object.entries(data.per_model).map(([model, m]) => `
    <div class="card" style="--card-color: ${MODEL_COLOR[model] || "var(--accent)"};">
      <div class="card-head"><span class="card-swatch"></span><span class="card-name">${model}</span></div>
      <div class="card-stats">
        <div class="stat"><span class="stat-value tabular">${m.mean_tok_s.toFixed(1)}</span><span class="stat-label">tok/s</span></div>
        <div class="stat"><span class="stat-value tabular">${m.mean_vram_mb.toFixed(0)}</span><span class="stat-label">VRAM MB</span></div>
        <div class="stat"><span class="stat-value tabular">${m.mean_quality.toFixed(2)}</span><span class="stat-label">quality /5</span></div>
      </div>
    </div>`).join("");
  return `<div class="section">
    <div class="section-title">Phase 3: Comparison</div>
    <div class="section-tag">${data.source_file}</div>
    <div class="cards">${cards}</div>
  </div>`;
}

async function loadDashboard() {
  const container = document.getElementById("dashboardContent");
  try {
    const [p1, p2, p3] = await Promise.all([
      fetch("/api/benchmark/phase1").then((r) => r.json()),
      fetch("/api/benchmark/phase2").then((r) => r.json()),
      fetch("/api/benchmark/phase3").then((r) => r.json()),
    ]);
    container.innerHTML = renderPhase1(p1) + renderPhase2(p2) + renderPhase3(p3);
    dashboardLoaded = true;
  } catch (err) {
    container.innerHTML = `<div class="unavailable">Failed to load benchmark results: ${err.message}</div>`;
  }
}

// ============ init ============
initTabs();
initChat();
loadModels();
