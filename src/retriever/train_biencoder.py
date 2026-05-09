import argparse
import json
import os
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.readers import InputExample
from torch.utils.data import DataLoader


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _project_root() -> str:
    # This file lives at src/retriever/train_biencoder.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abspath_from_root(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_project_root(), p))


def _build_ir_eval(examples: List[Dict[str, Any]]) -> InformationRetrievalEvaluator:
    # Build a lightweight IR evaluator from (query -> positive_doc_id) pairs.
    # This uses passage texts as corpus, keyed by doc_id.
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, set] = {}
    corpus: Dict[str, str] = {}

    # Build corpus from unique doc_ids referenced by positives/negatives.
    # The training jsonl stores doc_ids; we need to resolve them via passages.jsonl.
    passages = _read_jsonl(_abspath_from_root("data/kb/passages.jsonl"))
    for p in passages:
        corpus[p["doc_id"]] = p["passage_text"]

    for i, ex in enumerate(examples[:1000]):
        qid = str(i)
        queries[qid] = ex["query"]
        relevant_docs[qid] = {ex["positive_doc_id"]}

    return InformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=relevant_docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/retriever_train.jsonl")
    ap.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="models/retriever")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    args = ap.parse_args()

    train_path = _abspath_from_root(args.train)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n\n"
            "You need to generate it first by running notebooks/01_data_prep.ipynb (Section 3.2), "
            "which writes data/processed/retriever_train.jsonl."
        )

    train_rows = _read_jsonl(train_path)
    if not train_rows:
        raise RuntimeError(f"No training rows found at {train_path}")

    # Resolve positive_doc_id -> passage_text
    passages = _read_jsonl(_abspath_from_root("data/kb/passages.jsonl"))
    passage_by_id = {p["doc_id"]: p["passage_text"] for p in passages}

    train_examples: List[InputExample] = []
    for r in train_rows:
        pos_text = passage_by_id.get(r["positive_doc_id"])
        if not pos_text:
            continue
        train_examples.append(InputExample(texts=[r["query"], pos_text]))

    model = SentenceTransformer(args.base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    evaluator = _build_ir_eval(train_rows)

    # ----------------- TRAINING LOOP (SentenceTransformers fit) -----------------
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=max(1000, len(train_dataloader)),
        output_path=_abspath_from_root(args.out),
        save_best_model=True,
        show_progress_bar=True,
    )

    # ⏸️ TRAINING CHECKPOINT — Run the above code on Lightning AI before continuing.
    # Expected time: ~20–60 minutes on a single T4/A10 GPU for the filtered dataset size.
    # Save: the trained retriever to models/retriever/ (created by output_path).

    # ----------------- QUICK EVAL (reload + evaluator) -----------------
    tuned = SentenceTransformer(_abspath_from_root(args.out))
    score = evaluator(tuned)
    print("IR eval (subset) score:", score)


if __name__ == "__main__":
    main()
