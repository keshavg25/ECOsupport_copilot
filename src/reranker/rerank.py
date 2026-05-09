from __future__ import annotations

import os
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_path: str = "models/reranker") -> None:
        self.model = CrossEncoder(model_path)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, c.get("passage_text") or c.get("text") or ""] for c in candidates]
        scores = self.model.predict(pairs)

        out: List[Dict[str, Any]] = []
        for c, s in zip(candidates, scores):
            c2 = dict(c)
            c2["rerank_score"] = float(s)
            out.append(c2)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k]


def load_reranker() -> Reranker:
    return Reranker()
