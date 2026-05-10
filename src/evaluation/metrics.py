import re
from typing import Any, Dict, List, Tuple


_CITATION_RE = re.compile(r"\[(DOC_[0-9]+|POLICY:[^\]]+|TICKET)\]")


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

    cited = set(re.findall(r"\[(DOC_[0-9]+)\]", answer or ""))
    retrieved_set = set(retrieved)

    if not retrieved_set:
        return {"retrieved": 0, "cited_retrieved": 0, "precision": None}

    cited_retrieved = len(retrieved_set.intersection(cited))
    precision = cited_retrieved / max(1, len(cited)) if cited else 0.0

    return {
        "retrieved": len(retrieved_set),
        "cited_retrieved": cited_retrieved,
        "citation_precision": precision,
    }


def compute_all(answer: str, tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(citation_coverage(answer))
    out.update(tool_use_stats(tool_trace))
    out.update(retrieval_overlap(tool_trace, answer))
    out["answer_chars"] = len(answer or "")
    return out
