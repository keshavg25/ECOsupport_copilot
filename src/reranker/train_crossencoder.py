import argparse
import json
import os
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abspath_from_root(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_project_root(), p))


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
    ap.add_argument("--train", default="data/processed/reranker_train.jsonl")
    ap.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="models/reranker")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    train_path = _abspath_from_root(args.train)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n\n"
            "Generate it first using the data prep step (Section 3.3) which writes data/processed/reranker_train.jsonl."
        )

    rows = _read_jsonl(train_path)
    if not rows:
        raise RuntimeError(f"No training rows found at {train_path}")

    # CrossEncoder expects: list of ([query, passage], label)
    train_samples = [([r["query"], r["passage"]], float(r["label"])) for r in rows]

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
    )

    out_dir = _abspath_from_root(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ----------------- TRAINING LOOP -----------------
    model.fit(
        train_dataloader=train_samples,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        batch_size=args.batch_size,
        show_progress_bar=True,
        output_path=out_dir,
    )

    # ⏸️ TRAINING CHECKPOINT — Run the above code on Lightning AI before continuing.
    # Expected time: ~10–25 minutes on a single L4 GPU (depends on dataset size).
    # Save: models/reranker/ (created by output_path).

    # Quick smoke test
    scores = model.predict([["return policy", "You may return within 30 days."], ["return policy", "This is about batteries."]])
    print("Smoke scores:", scores)


if __name__ == "__main__":
    main()
