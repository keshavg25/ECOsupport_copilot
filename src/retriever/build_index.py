import argparse
import json
import os
import pickle
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="data/kb/passages.jsonl")
    ap.add_argument("--model", default="models/retriever")
    ap.add_argument("--out_index", default="data/kb/faiss.index")
    ap.add_argument("--out_map", default="data/kb/docid_map.pkl")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    passages_path = os.path.abspath(args.passages)
    rows = _read_jsonl(passages_path)
    if not rows:
        raise RuntimeError(f"No passages found at {passages_path}")

    model = SentenceTransformer(args.model)

    texts = [r["passage_text"] for r in rows]
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    faiss.write_index(index, args.out_index)

    with open(args.out_map, "wb") as f:
        pickle.dump(rows, f)

    print(f"Saved FAISS index to {args.out_index} (ntotal={index.ntotal}, dim={dim})")
    print(f"Saved doc map to {args.out_map} (records={len(rows)})")


if __name__ == "__main__":
    main()
