import os
import re
import gc
import json
import random
import torch
import numpy as np

from typing import Any, Dict, List
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# ============================================================
# GPU OPTIMIZATION
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device}")

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# DIRECTORIES
# ============================================================

ROOT = os.getcwd()

DATA_DIR = os.path.join(ROOT, "data")
KB_DIR = os.path.join(DATA_DIR, "kb")
PROC_DIR = os.path.join(DATA_DIR, "processed")
TOOL_DIR = os.path.join(DATA_DIR, "synthetic_tool_labels")

os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(TOOL_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

MAX_PASSAGES = 15000
MAX_QA = 5000
NEGATIVES_PER_QUERY = 5
BM25_CANDIDATES = 50
MAX_DPO_EXAMPLES = 200
MAX_NEW_TOKENS = 128

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2-7B-Instruct"

# ============================================================
# HELPERS
# ============================================================

def save_jsonl(path: str, rows: List[Dict[str, Any]]):

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str):

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def clean_text(x: str):

    x = str(x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

bitext = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

msmarco = load_dataset(
    "microsoft/ms_marco",
    "v2.1"
)

try:

    faq = load_dataset("Bellerophon/amazon-faq-dataset")
    faq_rows = faq["train"]

except Exception:

    faq_rows = []

# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

print("\nBuilding KB passages...")

passages = []

doc_id = 0

for row in tqdm(bitext["train"]):

    q = clean_text(row.get("instruction", ""))
    a = clean_text(row.get("response", ""))

    if len(q) < 5 or len(a) < 5:
        continue

    passages.append({
        "doc_id": f"DOC_{doc_id}",
        "title": q[:80],
        "passage_text": a,
        "source": "bitext"
    })

    doc_id += 1

    if len(passages) >= MAX_PASSAGES:
        break

for row in tqdm(faq_rows):

    q = clean_text(row.get("question", ""))
    a = clean_text(row.get("answer", ""))

    if len(q) < 5 or len(a) < 5:
        continue

    passages.append({
        "doc_id": f"DOC_{doc_id}",
        "title": q[:80],
        "passage_text": a,
        "source": "faq"
    })

    doc_id += 1

    if len(passages) >= MAX_PASSAGES:
        break

print(f"\nTotal passages: {len(passages)}")

# SAVE PASSAGES

passages_path = os.path.join(
    KB_DIR,
    "passages.jsonl"
)

save_jsonl(passages_path, passages)

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    EMBED_MODEL,
    device=device
)

passage_texts = [
    p["passage_text"]
    for p in passages
]

# ============================================================
# EMBEDDINGS
# ============================================================

print("\nGenerating embeddings on GPU...")

passage_embeddings = embedder.encode(
    passage_texts,
    batch_size=128,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ============================================================
# BM25 INDEX
# ============================================================

print("\nBuilding BM25 index...")

tokenized_corpus = [
    p.lower().split()
    for p in passage_texts
]

bm25 = BM25Okapi(tokenized_corpus)

# ============================================================
# RETRIEVER TRAIN DATA
# ============================================================

print("\nGenerating retriever training data...")

retriever_rows = []

for idx, row in enumerate(tqdm(bitext["train"])):

    if idx >= MAX_QA:
        break

    query = clean_text(row.get("instruction", ""))
    answer = clean_text(row.get("response", ""))

    if len(query) < 5:
        continue

    scores = bm25.get_scores(
        query.lower().split()
    )

    ranked_idx = np.argsort(scores)[::-1]

    positive_idx = ranked_idx[0]

    negative_idx = ranked_idx[
        1:NEGATIVES_PER_QUERY + 1
    ]

    retriever_rows.append({
        "query": query,
        "positive_doc_id": passages[positive_idx]["doc_id"],
        "negative_doc_ids": [
            passages[i]["doc_id"]
            for i in negative_idx
        ]
    })

print(f"\nRetriever samples: {len(retriever_rows)}")

save_jsonl(
    os.path.join(PROC_DIR, "retriever_train.jsonl"),
    retriever_rows
)

# ============================================================
# RERANKER TRAIN DATA
# ============================================================

print("\nGenerating reranker training data...")

by_doc = {
    p["doc_id"]: p
    for p in passages
}

reranker_rows = []

for row in tqdm(retriever_rows):

    query = row["query"]

    pos_doc = by_doc[
        row["positive_doc_id"]
    ]["passage_text"]

    reranker_rows.append({
        "query": query,
        "passage": pos_doc,
        "label": 1
    })

    for neg_id in row["negative_doc_ids"]:

        neg_doc = by_doc[
            neg_id
        ]["passage_text"]

        reranker_rows.append({
            "query": query,
            "passage": neg_doc,
            "label": 0
        })

save_jsonl(
    os.path.join(PROC_DIR, "reranker_train.jsonl"),
    reranker_rows
)

# ============================================================
# GENERATOR DATA
# ============================================================

print("\nGenerating instruction tuning data...")

SYSTEM_PROMPT = (
    "You are EcoSupport-Copilot. "
    "Answer using evidence from the KB. "
    "Always cite document ids. "
    "Escalate when unsure."
)

def build_context(row):

    ids = [
        row["positive_doc_id"]
    ] + row["negative_doc_ids"][:2]

    blocks = []

    for did in ids:

        blocks.append(
            f"[{did}] "
            f"{by_doc[did]['passage_text']}"
        )

    return "\n".join(blocks)

generator_rows = []

for row in tqdm(retriever_rows):

    context = build_context(row)

    generator_rows.append({
        "system": SYSTEM_PROMPT,
        "user": row["query"],
        "context": context,
        "assistant": (
            "Based on the retrieved evidence, "
            "here is the grounded answer."
        )
    })

save_jsonl(
    os.path.join(PROC_DIR, "generator_train.jsonl"),
    generator_rows
)

# ============================================================
# LOAD GENERATION MODEL
# ============================================================

print("\nLoading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(
    GEN_MODEL
)

model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============================================================
# DPO DATA
# ============================================================

print("\nGenerating DPO pairs...")

def make_prompt(example):

    return f"""
System:
{example['system']}

Context:
{example['context']}

User:
{example['user']}

Assistant:
"""

dpo_rows = []

for ex in tqdm(generator_rows[:MAX_DPO_EXAMPLES]):

    prompt = make_prompt(ex)

    output = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True
    )[0]["generated_text"]

    rejected = output[len(prompt):]

    chosen = ex["assistant"]

    dpo_rows.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

save_jsonl(
    os.path.join(PROC_DIR, "dpo_pairs.jsonl"),
    dpo_rows
)

# ============================================================
# TOOL LABELS
# ============================================================

print("\nGenerating tool labels...")

def tool_policy(query):

    q = query.lower()

    if any(
        x in q
        for x in [
            "refund",
            "return",
            "exchange",
            "warranty"
        ]
    ):

        return {
            "tool_name": "policy_lookup",
            "arguments": {
                "section": "RETURN_POLICY"
            }
        }

    if any(
        x in q
        for x in [
            "issue",
            "complaint",
            "ticket",
            "problem"
        ]
    ):

        return {
            "tool_name": "ticket_create",
            "arguments": {
                "priority": "medium"
            }
        }

    return {
        "tool_name": "search_kb",
        "arguments": {
            "top_k": 5
        }
    }

tool_rows = []

for row in tqdm(retriever_rows):

    tool_rows.append({
        "query": row["query"],
        "tool_call": tool_policy(
            row["query"]
        )
    })

save_jsonl(
    os.path.join(
        TOOL_DIR,
        "tool_train.jsonl"
    ),
    tool_rows
)

# ============================================================
# CLEANUP
# ============================================================

print("\nCleaning GPU memory...")

del model
del pipe

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
# DONE
# ============================================================

print("\nAll preprocessing completed successfully.")

print("\nGenerated files:")

print("data/kb/passages.jsonl")
print("data/processed/retriever_train.jsonl")
print("data/processed/reranker_train.jsonl")
print("data/processed/generator_train.jsonl")
print("data/processed/dpo_pairs.jsonl")
print("data/synthetic_tool_labels/tool_train.jsonl")