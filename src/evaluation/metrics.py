import re
from typing import Any, Dict, List, Optional

from rouge_score import rouge_scorer


_CITATION_RE = re.compile(r"\[(DOC_[0-9]+(?:@(?:[0-9]+-[0-9]+|p[0-9]+))?|POLICY:[^\]]+|TICKET)\]")


def citation_coverage(answer: str) -> Dict[str, Any]:
    cites = _CITATION_RE.findall(answer or "")
    return {
        "num_citations": len(cites),
        "has_citation": bool(cites),
        "unique_citations": len(set(cites)),
    }


def tool_use_stats(tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    names = [t.get("name") for t in (tool_trace or []) if isinstance(t, dict)]
    return {
        "num_tool_calls": len(names),
        "tools": names,
        "used_search": "SearchKB" in names,
        "used_policy": "GetPolicy" in names,
        "used_ticket": "CreateTicket" in names,
    }


def _evidence_text(tool_trace: List[Dict[str, Any]], *, max_chars: int = 12000) -> str:
    parts: List[str] = []
    for step in tool_trace or []:
        name = step.get("name")
        output = step.get("output") or {}
        if name == "SearchKB":
            for p in output.get("passages", []) or []:
                doc_id = p.get("doc_id")
                text = (p.get("text") or "").strip().replace("\n", " ")
                if doc_id:
                    parts.append(f"[{doc_id}] {text}")
                else:
                    parts.append(text)
        elif name == "GetPolicy":
            policy_text = (output.get("policy_text") or "").strip()
            if policy_text:
                parts.append(policy_text)
        elif name == "CreateTicket":
            # Tickets are not evidence for grounding.
            continue

    evidence = "\n".join(parts)
    return evidence[:max_chars]


def retrieval_overlap(tool_trace: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    """Heuristic faithfulness-ish metric:

    - Extract doc_ids returned by SearchKB.
    - Count how many of those doc_ids were cited in the answer.
    """

    retrieved: List[str] = []
    for step in tool_trace or []:
        if step.get("name") != "SearchKB":
            continue
        for p in (step.get("output") or {}).get("passages", []):
            doc_id = p.get("doc_id")
            if doc_id:
                retrieved.append(str(doc_id))

    # Accept [DOC_12], [DOC_12@start-end], [DOC_12@pN] and normalize to doc_id.
    cited = set(re.findall(r"\[(DOC_[0-9]+)(?:@(?:[0-9]+-[0-9]+|p[0-9]+))?\]", answer or ""))
    retrieved_set = set(retrieved)

    if not retrieved_set:
        return {"retrieved": 0, "cited_retrieved": 0, "citation_precision": None, "citation_recall": None}

    cited_retrieved = len(retrieved_set.intersection(cited))
    precision = cited_retrieved / max(1, len(cited)) if cited else 0.0

    recall = cited_retrieved / max(1, len(retrieved_set))

    return {
        "retrieved": len(retrieved_set),
        "cited_retrieved": cited_retrieved,
        "citation_precision": precision,
        "citation_recall": recall,
    }


def grounding_rouge_l(answer: str, tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grounding metric: ROUGE-L fmeasure(answer, evidence).

    This is a proxy for "answer uses retrieved evidence".
    """

    ans = (answer or "").strip()
    ev = _evidence_text(tool_trace)
    if not ans or not ev:
        return {"grounding_rougeL_f": None}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(ev, ans)["rougeL"].fmeasure
    return {"grounding_rougeL_f": float(score)}


def compute_all(answer: str, tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(citation_coverage(answer))
    out.update(tool_use_stats(tool_trace))
    out.update(retrieval_overlap(tool_trace, answer))
    out.update(grounding_rouge_l(answer, tool_trace))
    out["answer_chars"] = len(answer or "")
    return out
