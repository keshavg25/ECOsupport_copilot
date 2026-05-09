import argparse
import json
import os
import re
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _label_tool(query: str) -> Dict[str, Any]:
    q = (query or "").lower()

    # Heuristics from spec
    if any(k in q for k in ["policy", "return", "refund", "exchange", "warranty", "shipping", "delivery"]):
        # Choose a section_id based on keyword
        if any(k in q for k in ["return", "refund"]):
            section_id = "RETURN_POLICY"
        elif "exchange" in q:
            section_id = "EXCHANGE_POLICY"
        elif "warranty" in q:
            section_id = "WARRANTY_POLICY"
        elif any(k in q for k in ["ship", "shipping"]):
            section_id = "SHIPPING_POLICY"
        elif any(k in q for k in ["deliver", "delivery", "missing", "lost"]):
            section_id = "DELIVERY_ISSUES_POLICY"
        else:
            section_id = "RETURN_POLICY"

        return {
            "name": "GetPolicy",
            "args": {"section_id": section_id},
            "reason": "Query mentions a policy-like intent; fetch authoritative policy text.",
        }

    if any(k in q for k in ["complaint", "not resolved", "escalate", "angry", "chargeback", "fraud"]):
        return {
            "name": "CreateTicket",
            "args": {
                "summary": query.strip()[:240],
                "category": "support",
                "severity": "high" if any(k in q for k in ["fraud", "chargeback"]) else "medium",
            },
            "reason": "User indicates unresolved complaint/escalation; open a ticket.",
        }

    if re.search(r"\b(hi|hello|hey|thanks)\b", q) and len(q.split()) <= 5:
        return {"name": "None", "args": {}, "reason": "Greeting/simple message; no tool needed."}

    # default open-ended product / support question
    return {
        "name": "SearchKB",
        "args": {"query": query},
        "reason": "Open-ended question; retrieve relevant KB passages.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL with {query, passages_context}? or {query}")
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSONL path (e.g., data/synthetic_tool_labels/tool_train.jsonl)",
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.input)
    out: List[Dict[str, Any]] = []

    for r in rows:
        query = r.get("query") or r.get("question") or ""
        passages_context = r.get("passages_context") or r.get("passage_block") or r.get("passages") or ""

        labeled = _label_tool(query)
        out.append(
            {
                "query": query,
                "passages_context": passages_context,
                "tool_call": {"name": labeled["name"], "args": labeled.get("args", {})},
                "reason": labeled.get("reason", ""),
            }
        )

    _write_jsonl(args.output, out)
    print(f"Wrote {len(out)} labeled examples to {args.output}")


if __name__ == "__main__":
    main()
