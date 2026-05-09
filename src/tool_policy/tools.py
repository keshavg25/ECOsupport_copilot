import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def SearchKB(
    query: str,
    faiss_index: Any,
    doc_map: List[Dict[str, Any]],
    encoder: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Returns: {passages: [{doc_id, text, score, span_start, span_end}]}"""
    if not query or not isinstance(query, str):
        return {"passages": []}

    q_emb = encoder.encode([query], normalize_embeddings=True)
    scores, idxs = faiss_index.search(q_emb, top_k)

    passages: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(doc_map):
            continue
        rec = doc_map[idx]
        passages.append(
            {
                "doc_id": rec["doc_id"],
                "text": rec["passage_text"],
                "score": float(score),
                "span_start": int(rec.get("span_start", 0)),
                "span_end": int(rec.get("span_end", 0)),
                "category": rec.get("category"),
            }
        )

    return {"passages": passages}


def GetPolicy(section_id: str, policy_db: Dict[str, Any]) -> Dict[str, Any]:
    """Returns: {section_id, policy_text, last_updated}"""
    if not section_id:
        return {"section_id": "", "policy_text": "", "last_updated": None}

    rec = policy_db.get(section_id)
    if not rec:
        return {
            "section_id": section_id,
            "policy_text": "Policy section not found.",
            "last_updated": None,
        }

    return {
        "section_id": rec.get("section_id", section_id),
        "policy_text": rec.get("policy_text", ""),
        "last_updated": rec.get("last_updated"),
        "title": rec.get("title"),
    }


def CreateTicket(summary: str, category: str, severity: str) -> Dict[str, Any]:
    """Returns: {ticket_id, status, estimated_resolution_hours}

    Mock implementation: generates UUID ticket_id and logs to data/tickets.jsonl
    """
    ticket_id = str(uuid.uuid4())
    record = {
        "ticket_id": ticket_id,
        "created_at": _now_iso(),
        "summary": summary,
        "category": category,
        "severity": severity,
        "status": "open",
    }

    # heuristic ETA
    sev = (severity or "").lower().strip()
    if sev in {"critical", "sev0", "sev1", "high"}:
        eta = 4
    elif sev in {"medium", "sev2"}:
        eta = 24
    else:
        eta = 48

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "tickets.jsonl")

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "estimated_resolution_hours": eta,
    }
