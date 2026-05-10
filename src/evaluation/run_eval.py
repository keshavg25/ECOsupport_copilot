import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Ensure `src.*` imports work when executed as a script.
_ROOT = _project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from src.evaluation.metrics import compute_all
from src.pipeline.copilot import EcoSupportCopilot


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _default_questions() -> List[Dict[str, Any]]:
    return [
        {"id": "q1", "question": "What is your return policy for unopened items?"},
        {"id": "q2", "question": "How do I contact customer support?"},
        {"id": "q3", "question": "I was double-charged. What should I do?"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", default="")
    ap.add_argument("--out_csv", default="artifacts/eval_results.csv")

    ap.add_argument("--tool_policy_base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tool_policy_adapter", default="models/tool_policy")

    ap.add_argument("--generator_base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--generator_adapter", default="models/generator_dpo")

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    args = ap.parse_args()

    if args.questions_jsonl:
        questions = _read_jsonl(args.questions_jsonl)
    else:
        questions = _default_questions()

    copilot = EcoSupportCopilot(
        tool_policy_base=args.tool_policy_base,
        tool_policy_adapter=args.tool_policy_adapter,
        generator_base=args.generator_base,
        generator_adapter=args.generator_adapter,
        top_k_default=args.top_k,
    ).load()

    rows: List[Dict[str, Any]] = []
    for ex in questions:
        qid = ex.get("id", "")
        q = ex.get("question") or ex.get("query")
        if not q:
            continue

        answer, trace = copilot.answer(q, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
        metrics = compute_all(answer, trace)

        rows.append(
            {
                "id": qid,
                "question": q,
                "answer": answer,
                "tool_trace": json.dumps(trace, ensure_ascii=False),
                **metrics,
            }
        )

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {
        "n": len(df),
        "has_citation_rate": float(df["has_citation"].mean()) if len(df) else 0.0,
        "avg_num_citations": float(df["num_citations"].mean()) if len(df) else 0.0,
        "used_search_rate": float(df["used_search"].mean()) if len(df) else 0.0,
    }

    print("=== EVAL_SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
