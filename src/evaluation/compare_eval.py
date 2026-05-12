import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def _run(
    questions: List[Dict[str, Any]],
    *,
    tool_policy_base: str,
    tool_policy_adapter: str,
    generator_base: str,
    generator_adapter: str,
    top_k: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    copilot = EcoSupportCopilot(
        tool_policy_base=tool_policy_base,
        tool_policy_adapter=tool_policy_adapter,
        generator_base=generator_base,
        generator_adapter=generator_adapter,
        top_k_default=top_k,
    ).load()

    rows: List[Dict[str, Any]] = []
    for ex in questions:
        qid = ex.get("id", "")
        q = ex.get("question") or ex.get("query")
        if not q:
            continue

        answer, trace = copilot.answer(q, top_k=top_k, max_new_tokens=max_new_tokens)
        metrics = compute_all(answer, trace)
        rows.append({"id": qid, "question": q, "answer": answer, "tool_trace": trace, **metrics})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", default="")
    ap.add_argument("--out_csv", default="artifacts/compare_eval.csv")

    ap.add_argument("--tool_policy_base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--generator_base", default="Qwen/Qwen2.5-1.5B-Instruct")

    ap.add_argument("--baseline_tool_policy_adapter", default="none")
    ap.add_argument("--baseline_generator_adapter", default="none")

    ap.add_argument("--advanced_tool_policy_adapter", default="models/tool_policy")
    ap.add_argument("--advanced_generator_adapter", default="models/generator_dpo")

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    args = ap.parse_args()

    questions = _read_jsonl(args.questions_jsonl) if args.questions_jsonl else _default_questions()

    df_base = _run(
        questions,
        tool_policy_base=args.tool_policy_base,
        tool_policy_adapter=args.baseline_tool_policy_adapter,
        generator_base=args.generator_base,
        generator_adapter=args.baseline_generator_adapter,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    df_adv = _run(
        questions,
        tool_policy_base=args.tool_policy_base,
        tool_policy_adapter=args.advanced_tool_policy_adapter,
        generator_base=args.generator_base,
        generator_adapter=args.advanced_generator_adapter,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    # Join and compute deltas for numeric metrics.
    key_cols = ["id", "question"]
    df_base = df_base.add_prefix("baseline_")
    df_adv = df_adv.add_prefix("advanced_")

    df = pd.concat(
        [
            df_base[["baseline_id", "baseline_question"]].rename(columns={"baseline_id": "id", "baseline_question": "question"}),
            df_base.drop(columns=["baseline_id", "baseline_question"]),
            df_adv.drop(columns=["advanced_id", "advanced_question"]),
        ],
        axis=1,
    )

    # delta columns
    numeric_candidates = [
        "num_citations",
        "unique_citations",
        "citation_precision",
        "citation_recall",
        "grounding_rougeL_f",
        "answer_chars",
        "num_tool_calls",
        "retrieved",
        "cited_retrieved",
    ]
    for name in numeric_candidates:
        b = f"baseline_{name}"
        a = f"advanced_{name}"
        if b in df.columns and a in df.columns:
            df[f"delta_{name}"] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Summary
    def _mean(col: str) -> float:
        s = pd.to_numeric(df[col], errors="coerce")
        return float(s.mean()) if len(s.dropna()) else 0.0

    summary = {
        "n": int(len(df)),
        "baseline_grounding_rougeL_f": _mean("baseline_grounding_rougeL_f"),
        "advanced_grounding_rougeL_f": _mean("advanced_grounding_rougeL_f"),
        "delta_grounding_rougeL_f": _mean("delta_grounding_rougeL_f"),
        "baseline_citation_recall": _mean("baseline_citation_recall"),
        "advanced_citation_recall": _mean("advanced_citation_recall"),
        "delta_citation_recall": _mean("delta_citation_recall"),
    }

    print("=== COMPARE_EVAL_SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
