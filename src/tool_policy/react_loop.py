import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Ensure `src.*` imports work when executed as a script.
_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _abspath_from_root(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_project_root(), p))


def _load_policy_db() -> Dict[str, Any]:
    path = _abspath_from_root("data/kb/policies.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_faiss() -> Tuple[Any, List[Dict[str, Any]]]:
    import faiss

    index_path = _abspath_from_root("data/kb/faiss.index")
    map_path = _abspath_from_root("data/kb/docid_map.pkl")

    index = faiss.read_index(index_path)

    import pickle

    with open(map_path, "rb") as f:
        doc_map = pickle.load(f)

    return index, doc_map


def _load_retriever_encoder() -> Any:
    from sentence_transformers import SentenceTransformer

    model_path = _abspath_from_root("models/retriever")
    return SentenceTransformer(model_path)


def _load_reranker() -> Any:
    from sentence_transformers import CrossEncoder

    path = _abspath_from_root("models/reranker")
    return CrossEncoder(path)


def _is_disabled_adapter(path: Optional[str]) -> bool:
    if path is None:
        return True
    p = str(path).strip().lower()
    return p in {"", "none", "null", "no", "false"}


def _load_tool_policy_model(base_model: str, adapter_dir: Optional[str]) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if not _is_disabled_adapter(adapter_dir):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def _load_generator(base_model: str, adapter_dir: Optional[str]) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if not _is_disabled_adapter(adapter_dir):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try to find the first JSON object in the output
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_json_arg(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _generate(model: Any, tok: Any, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    # Return tail after prompt when possible
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def _ensure_doc_citations(answer: str, retrieved_doc_ids: List[str], max_ids: int = 3) -> str:
    ans = (answer or "").strip()
    if re.search(r"\[DOC_[0-9]+\]", ans):
        return ans
    doc_ids = [d for d in retrieved_doc_ids if isinstance(d, str) and d.startswith("DOC_")]
    doc_ids = list(dict.fromkeys(doc_ids))[:max_ids]
    if not doc_ids:
        return ans
    sources = " ".join(f"[{d}]" for d in doc_ids)
    if ans:
        return f"{ans}\n\nSources: {sources}".strip()
    return f"Sources: {sources}".strip()


@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    output: Dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)

    ap.add_argument("--tool_policy_base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument(
        "--tool_policy_adapter",
        default="models/tool_policy",
        help="Adapter path, or 'none' to disable (baseline).",
    )

    ap.add_argument("--generator_base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--generator_adapter",
        default="models/generator_dpo",
        help="Adapter path, or 'none' to disable (baseline).",
    )

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument(
        "--force_tool",
        default="",
        help="Force tool name (SearchKB, GetPolicy, CreateTicket, None). If set, bypasses tool-policy model.",
    )
    ap.add_argument(
        "--force_tool_args",
        default="",
        help='Optional JSON dict for forced tool args. Example: --force_tool_args "{\"query\":\"...\",\"top_k\":5}"',
    )
    args = ap.parse_args()

    # Load resources
    from src.tool_policy.tools import CreateTicket, GetPolicy, SearchKB

    policy_db = _load_policy_db()
    faiss_index, doc_map = _load_faiss()
    encoder = _load_retriever_encoder()
    reranker = _load_reranker()

    tool_policy_adapter = None if _is_disabled_adapter(args.tool_policy_adapter) else _abspath_from_root(args.tool_policy_adapter)
    tool_model, tool_tok = _load_tool_policy_model(args.tool_policy_base, tool_policy_adapter)

    generator_adapter = None if _is_disabled_adapter(args.generator_adapter) else _abspath_from_root(args.generator_adapter)
    gen_model, gen_tok = _load_generator(args.generator_base, generator_adapter)

    # 1) Decide tool (or force tool)
    forced_name = (args.force_tool or "").strip()
    forced_args = _parse_json_arg(args.force_tool_args)
    if forced_name:
        tool_call = {"name": forced_name, "args": forced_args}
    else:
        tool_prompt = (
            "Decide one tool call as JSON: {\"name\": string, \"args\": object}. "
            "Allowed: SearchKB, GetPolicy, CreateTicket, None.\n\n"
            f"User query: {args.question}\n"
        )
        tool_out = _generate(tool_model, tool_tok, tool_prompt, max_new_tokens=128)
        tool_call = _extract_json(tool_out) or {"name": "SearchKB", "args": {"query": args.question, "top_k": args.top_k}}

    name = tool_call.get("name") or tool_call.get("tool") or "SearchKB"
    call_args = tool_call.get("args") or {}

    tool_trace: List[ToolResult] = []

    # 2) Execute tool
    if name == "None":
        pass
    elif name == "GetPolicy":
        section_id = call_args.get("section_id", "")
        out = GetPolicy(section_id=section_id, policy_db=policy_db)
        tool_trace.append(ToolResult(name=name, args={"section_id": section_id}, output=out))
    elif name == "CreateTicket":
        out = CreateTicket(
            summary=str(call_args.get("summary", ""))[:240],
            category=str(call_args.get("category", "support")),
            severity=str(call_args.get("severity", "medium")),
        )
        tool_trace.append(ToolResult(name=name, args=call_args, output=out))
    else:
        q = call_args.get("query") or args.question
        top_k = int(call_args.get("top_k") or args.top_k)
        out = SearchKB(query=q, faiss_index=faiss_index, doc_map=doc_map, encoder=encoder, top_k=top_k)

        # Rerank passages
        passages = out.get("passages", [])
        pairs = [[args.question, p.get("text", "")] for p in passages]
        if pairs:
            scores = reranker.predict(pairs)
            order = list(np.argsort(-np.array(scores)))
            passages = [passages[i] for i in order]
            out["passages"] = passages
        tool_trace.append(ToolResult(name="SearchKB", args={"query": q, "top_k": top_k}, output=out))

    # 3) Build evidence block
    evidence_lines: List[str] = []
    retrieved_doc_ids: List[str] = []
    for tr in tool_trace:
        if tr.name == "SearchKB":
            for p in tr.output.get("passages", []):
                doc_id = p.get("doc_id")
                if doc_id:
                    retrieved_doc_ids.append(str(doc_id))
                text = (p.get("text") or "").strip().replace("\n", " ")
                evidence_lines.append(f"[{doc_id}] {text}")
        elif tr.name == "GetPolicy":
            evidence_lines.append(f"[POLICY:{tr.args.get('section_id')}] {tr.output.get('policy_text','')}")
        elif tr.name == "CreateTicket":
            evidence_lines.append(f"[TICKET] {json.dumps(tr.output, ensure_ascii=False)}")

    evidence = "\n".join(evidence_lines)[:12000]

    # 4) Generate final answer
    gen_prompt = (
        "System:\nYou are EcoSupport-Copilot. Answer using evidence from the KB/policies/tools. "
        "Always cite document ids in square brackets like [DOC_123]. If you cannot answer from evidence, escalate.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"User:\n{args.question}\n\n"
        "Assistant:\n"
    )
    answer = _generate(gen_model, gen_tok, gen_prompt, max_new_tokens=args.max_new_tokens)
    answer = _ensure_doc_citations(answer, retrieved_doc_ids)

    # 5) Emit response + tool trace JSON
    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== TOOL_TRACE_JSON ===\n")
    trace_json = [
        {"name": tr.name, "args": tr.args, "output": tr.output}
        for tr in tool_trace
    ]
    print(json.dumps(trace_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
