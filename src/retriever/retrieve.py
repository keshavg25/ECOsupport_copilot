import json
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        model_path: str = "models/retriever",
        index_path: str = "data/kb/faiss.index",
        docmap_path: str = "data/kb/docid_map.pkl",
    ) -> None:
        self.encoder = SentenceTransformer(model_path)
        self.index = faiss.read_index(index_path)
        with open(docmap_path, "rb") as f:
            self.doc_map: List[Dict[str, Any]] = pickle.load(f)

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.encoder.encode([query], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def search(self, embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        scores, idxs = self.index.search(embedding, k)
        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.doc_map):
                continue
            rec = self.doc_map[idx]
            out.append(
                {
                    "doc_id": rec["doc_id"],
                    "score": float(score),
                    "passage_text": rec["passage_text"],
                    "span": (int(rec.get("span_start", 0)), int(rec.get("span_end", 0))),
                    "category": rec.get("category"),
                }
            )
        return out


def load_retriever() -> Retriever:
    return Retriever()
