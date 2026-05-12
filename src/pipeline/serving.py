import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from src.pipeline.copilot import EcoSupportCopilot
from src.evaluation.metrics import compute_all


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    max_new_tokens: int = Field(default=256, ge=32, le=1024)


class ChatResponse(BaseModel):
    answer: str
    tool_trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


app = FastAPI(title="EcoSupport-Copilot", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_copilot: Optional[EcoSupportCopilot] = None


@app.on_event("startup")
def _startup() -> None:
    global _copilot

    tool_policy_base = _env("TOOL_POLICY_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
    tool_policy_adapter = _env("TOOL_POLICY_ADAPTER", "models/tool_policy")

    generator_base = _env("GENERATOR_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
    generator_adapter = _env("GENERATOR_ADAPTER", "models/generator_dpo")

    top_k_default = int(_env("TOP_K_DEFAULT", "5"))

    _copilot = EcoSupportCopilot(
        tool_policy_base=tool_policy_base,
        tool_policy_adapter=tool_policy_adapter,
        generator_base=generator_base,
        generator_adapter=generator_adapter,
        top_k_default=top_k_default,
    ).load()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
        return """<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>EcoSupport-Copilot</title>
    </head>
    <body>
        <h1>EcoSupport-Copilot</h1>

        <form id=\"chat-form\">
            <label for=\"question\">Question</label><br />
            <textarea id=\"question\" rows=\"4\" cols=\"80\" required></textarea><br />

            <label for=\"top_k\">top_k</label>
            <input id=\"top_k\" type=\"number\" min=\"1\" max=\"20\" value=\"5\" />

            <label for=\"max_new_tokens\">max_new_tokens</label>
            <input id=\"max_new_tokens\" type=\"number\" min=\"32\" max=\"1024\" value=\"220\" />

            <button id=\"send\" type=\"submit\">Ask</button>
            <span id=\"status\"></span>
        </form>

        <h2>Answer</h2>
        <pre id=\"answer\"></pre>

        <h2>Tool trace</h2>
        <pre id=\"trace\"></pre>

        <h2>Metrics</h2>
        <pre id=\"metrics\"></pre>

        <p>API docs: <a href=\"/docs\">/docs</a></p>    

        <script>
            const form = document.getElementById('chat-form');
            const statusEl = document.getElementById('status');
            const answerEl = document.getElementById('answer');
            const traceEl = document.getElementById('trace');
            const metricsEl = document.getElementById('metrics');
            const sendBtn = document.getElementById('send');

            function setBusy(isBusy) {
                sendBtn.disabled = isBusy;
                statusEl.textContent = isBusy ? 'Thinking…' : '';
            }

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const question = document.getElementById('question').value;
                const top_k = parseInt(document.getElementById('top_k').value || '5', 10);
                const max_new_tokens = parseInt(document.getElementById('max_new_tokens').value || '220', 10);

                answerEl.textContent = '';
                traceEl.textContent = '';
                metricsEl.textContent = '';
                setBusy(true);

                try {
                    const resp = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question, top_k, max_new_tokens})
                    });
                    const data = await resp.json();

                    if (!resp.ok) {
                        throw new Error(data?.detail || resp.statusText);
                    }

                    answerEl.textContent = data.answer || '';
                    traceEl.textContent = JSON.stringify(data.tool_trace || [], null, 2);
                    metricsEl.textContent = JSON.stringify(data.metrics || {}, null, 2);
                } catch (err) {
                    answerEl.textContent = 'Error: ' + (err?.message || String(err));
                } finally {
                    setBusy(false);
                }
            });
        </script>
    </body>
</html>"""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _copilot is None:
        raise RuntimeError("Service not initialized")
    answer, trace = _copilot.answer(req.question, top_k=req.top_k, max_new_tokens=req.max_new_tokens)
    metrics = compute_all(answer, trace)
    return ChatResponse(answer=answer, tool_trace=trace, metrics=metrics)
