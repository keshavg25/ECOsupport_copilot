import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _abspath_from_root(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_project_root(), p))


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _is_disabled_adapter(path: Optional[str]) -> bool:
    if path is None:
        return True
    p = str(path).strip().lower()
    return p in {"", "none", "null", "no", "false"}


def _generate(model: Any, tok: Any, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def _format_doc_span_citation(doc_id: str, span_start: int, span_end: int) -> str:
    s = int(span_start or 0)
    e = int(span_end or 0)
    if s <= 0 and e <= 0:
        return f"[{doc_id}]"
    if e < s:
        s, e = e, s
    return f"[{doc_id}@{s}-{e}]"


def _ensure_doc_citations(answer: str, retrieved_citations: List[str], max_ids: int = 3) -> str:
    """Rubric-like enforcement: ensure final answer includes at least one DOC citation.

    Required format: [DOC_12@123-456] (span is preferred), but we also accept [DOC_12].
    """

    ans = (answer or "").strip()
    if re.search(r"\[DOC_[0-9]+(?:@[0-9]+-[0-9]+)?\]", ans):
        return ans
    cites = [c for c in retrieved_citations if isinstance(c, str) and c.startswith("[DOC_")]
    cites = list(dict.fromkeys(cites))[:max_ids]
    if not cites:
        return ans
    sources = " ".join(cites)
    if ans:
        return f"{ans}\n\nSources: {sources}".strip()
    return f"Sources: {sources}".strip()


@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    output: Dict[str, Any]


class EcoSupportCopilot:
    def __init__(
        self,
        tool_policy_base: str,
        tool_policy_adapter: str,
        generator_base: str,
        generator_adapter: str,
        top_k_default: int = 5,
    ) -> None:
        self.tool_policy_base = tool_policy_base
        self.generator_base = generator_base

        # Normalize adapter inputs BEFORE making absolute paths.
        self.tool_policy_adapter = None if _is_disabled_adapter(tool_policy_adapter) else _abspath_from_root(tool_policy_adapter)
        self.generator_adapter = None if _is_disabled_adapter(generator_adapter) else _abspath_from_root(generator_adapter)
        self.top_k_default = top_k_default

        self._policy_db: Optional[Dict[str, Any]] = None
        self._faiss_index: Any = None
        self._doc_map: Optional[List[Dict[str, Any]]] = None
        self._encoder: Any = None
        self._reranker: Any = None

        self._tool_model: Any = None
        self._tool_tok: Any = None

        self._gen_model: Any = None
        self._gen_tok: Any = None

    def load(self) -> "EcoSupportCopilot":
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # KB / policies
        with open(_abspath_from_root("data/kb/policies.json"), "r", encoding="utf-8") as f:
            self._policy_db = json.load(f)

        import faiss
        import pickle

        self._faiss_index = faiss.read_index(_abspath_from_root("data/kb/faiss.index"))
        with open(_abspath_from_root("data/kb/docid_map.pkl"), "rb") as f:
            self._doc_map = pickle.load(f)

        self._encoder = SentenceTransformer(_abspath_from_root("models/retriever"))
        self._reranker = CrossEncoder(_abspath_from_root("models/reranker"))

        # Tool policy
        self._tool_tok = AutoTokenizer.from_pretrained(self.tool_policy_base, use_fast=True)
        if self._tool_tok.pad_token is None:
            self._tool_tok.pad_token = self._tool_tok.eos_token
        base_tool_model = AutoModelForCausalLM.from_pretrained(self.tool_policy_base, device_map="auto")
        if _is_disabled_adapter(self.tool_policy_adapter):
            self._tool_model = base_tool_model
        else:
            from peft import PeftModel

            self._tool_model = PeftModel.from_pretrained(base_tool_model, self.tool_policy_adapter)
        self._tool_model.eval()

        # Generator
        self._gen_tok = AutoTokenizer.from_pretrained(self.generator_base, use_fast=True)
        if self._gen_tok.pad_token is None:
            self._gen_tok.pad_token = self._gen_tok.eos_token
        base_gen_model = AutoModelForCausalLM.from_pretrained(self.generator_base, device_map="auto")
        if _is_disabled_adapter(self.generator_adapter):
            self._gen_model = base_gen_model
        else:
            from peft import PeftModel

            self._gen_model = PeftModel.from_pretrained(base_gen_model, self.generator_adapter)
        self._gen_model.eval()

        return self

    def _require_loaded(self) -> None:
        if (
            self._policy_db is None
            or self._faiss_index is None
            or self._doc_map is None
            or self._encoder is None
            or self._reranker is None
            or self._tool_model is None
            or self._tool_tok is None
            or self._gen_model is None
            or self._gen_tok is None
        ):
            raise RuntimeError("EcoSupportCopilot not loaded; call .load() first")

    def answer(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 256,
        max_tool_steps: int = 2,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self._require_loaded()

        from src.tool_policy.tools import CreateTicket, GetPolicy, SearchKB

        top_k = int(top_k or self.top_k_default)

        tool_trace: List[ToolResult] = []
        last_tool_json: str = ""
        for _step in range(max(1, int(max_tool_steps))):
            tool_prompt = (
                "Decide ONE tool call as JSON: {\"name\": string, \"args\": object}. "
                "Allowed tools: SearchKB, GetPolicy, CreateTicket, None.\n"
                "Rules: If you already have enough evidence to answer, choose name=\"None\".\n\n"
                f"User query: {question}\n"
                + (f"Previous tool output (JSON): {last_tool_json}\n" if last_tool_json else "")
            )
            tool_out = _generate(self._tool_model, self._tool_tok, tool_prompt, max_new_tokens=128)
            tool_call = _extract_json(tool_out) or {"name": "SearchKB", "args": {"query": question, "top_k": top_k}}

            name = tool_call.get("name") or tool_call.get("tool") or "SearchKB"
            call_args = tool_call.get("args") or {}

            if name == "None":
                break
            if name == "GetPolicy":
                section_id = call_args.get("section_id", "")
                out = GetPolicy(section_id=section_id, policy_db=self._policy_db)
                tr = ToolResult(name=name, args={"section_id": section_id}, output=out)
                tool_trace.append(tr)
                last_tool_json = json.dumps(out, ensure_ascii=False)[:2000]
                continue
            if name == "CreateTicket":
                out = CreateTicket(
                    summary=str(call_args.get("summary", ""))[:240],
                    category=str(call_args.get("category", "support")),
                    severity=str(call_args.get("severity", "medium")),
                )
                tr = ToolResult(name=name, args=call_args, output=out)
                tool_trace.append(tr)
                last_tool_json = json.dumps(out, ensure_ascii=False)[:2000]
                continue

            q = call_args.get("query") or question
            k = int(call_args.get("top_k") or top_k)
            out = SearchKB(query=q, faiss_index=self._faiss_index, doc_map=self._doc_map, encoder=self._encoder, top_k=k)

            passages = out.get("passages", [])
            pairs = [[question, p.get("text", "")] for p in passages]
            if pairs:
                scores = self._reranker.predict(pairs)
                order = list(np.argsort(-np.array(scores)))
                passages = [passages[i] for i in order]
                out["passages"] = passages

            tr = ToolResult(name="SearchKB", args={"query": q, "top_k": k}, output=out)
            tool_trace.append(tr)
            last_tool_json = json.dumps({"passages": out.get("passages", [])[:2]}, ensure_ascii=False)[:2000]

        evidence_lines: List[str] = []
        retrieved_citations: List[str] = []
        for tr in tool_trace:
            if tr.name == "SearchKB":
                for p in tr.output.get("passages", []):
                    doc_id = p.get("doc_id")
                    if doc_id:
                        retrieved_citations.append(
                            _format_doc_span_citation(
                                str(doc_id),
                                int(p.get("span_start", 0) or 0),
                                int(p.get("span_end", 0) or 0),
                            )
                        )
                    text = (p.get("text") or "").strip().replace("\n", " ")
                    span_start = int(p.get("span_start", 0) or 0)
                    span_end = int(p.get("span_end", 0) or 0)
                    cite = _format_doc_span_citation(str(doc_id), span_start, span_end) if doc_id else ""
                    evidence_lines.append(f"{cite} {text}".strip())
            elif tr.name == "GetPolicy":
                evidence_lines.append(f"[POLICY:{tr.args.get('section_id')}] {tr.output.get('policy_text','')}")
            elif tr.name == "CreateTicket":
                evidence_lines.append(f"[TICKET] {json.dumps(tr.output, ensure_ascii=False)}")

        evidence = "\n".join(evidence_lines)[:12000]
        gen_prompt = (
            "System:\nYou are EcoSupport-Copilot. Answer using evidence from the KB/policies/tools. "
            "Always cite evidence as [DOC_123@start-end] (doc id + span). "
            "Policy citations may use [POLICY:section_id]. "
            "If you cannot answer from evidence, escalate.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"User:\n{question}\n\n"
            "Assistant:\n"
        )
        answer = _generate(self._gen_model, self._gen_tok, gen_prompt, max_new_tokens=max_new_tokens)
        answer = _ensure_doc_citations(answer, retrieved_citations)

        trace_json = [{"name": tr.name, "args": tr.args, "output": tr.output} for tr in tool_trace]
        return answer, trace_json
