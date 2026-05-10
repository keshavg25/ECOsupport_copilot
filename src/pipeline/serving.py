import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from src.pipeline.copilot import EcoSupportCopilot


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    max_new_tokens: int = Field(default=256, ge=32, le=1024)


class ChatResponse(BaseModel):
    answer: str
    tool_trace: List[Dict[str, Any]]


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


app = FastAPI(title="EcoSupport-Copilot", version="0.1")

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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _copilot is None:
        raise RuntimeError("Service not initialized")
    answer, trace = _copilot.answer(req.question, top_k=req.top_k, max_new_tokens=req.max_new_tokens)
    return ChatResponse(answer=answer, tool_trace=trace)
