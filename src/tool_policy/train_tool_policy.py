import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


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


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    allowed = set(inspect.signature(fn).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print(f"[tool-policy] {label} dropped keys:", dropped)
    return filtered


def _format_example(tokenizer: Any, query: str, tool_call: Dict[str, Any]) -> str:
    system = (
        "You are the EcoSupport-Copilot tool-policy model. "
        "Given a user query, select exactly one tool call to execute, or return null if no tool is needed. "
        "Respond ONLY with valid JSON.\n\n"
        "Allowed tools:\n"
        "- SearchKB: {\"query\": string, \"top_k\": int}\n"
        "- GetPolicy: {\"section_id\": string}\n"
        "- CreateTicket: {\"summary\": string, \"category\": string, \"severity\": string}\n"
        "- None: {}\n"
        "JSON schema:\n"
        "{\"name\": string, \"args\": object}"
    )

    # Normalize legacy schema if present
    if "tool_name" in tool_call and "arguments" in tool_call:
        name = tool_call.get("tool_name")
        args = tool_call.get("arguments") or {}
        if name == "search_kb":
            tool_call = {"name": "SearchKB", "args": {"query": query, **args}}
        elif name == "get_policy":
            tool_call = {"name": "GetPolicy", "args": args}
        elif name == "create_ticket":
            tool_call = {"name": "CreateTicket", "args": args}
        elif name in {"none", "null", None}:
            tool_call = {"name": "None", "args": {}}
        else:
            tool_call = {"name": "SearchKB", "args": {"query": query, "top_k": 5}}

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
        {"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{query}\n"
        f"<|assistant|>\n{json.dumps(tool_call, ensure_ascii=False)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synthetic_tool_labels/tool_train.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="models/tool_policy")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    train_path = _abspath_from_root(args.train)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n\n"
            "Generate it using src/tool_policy/generate_tool_labels.py or provide it explicitly."
        )

    out_dir = _abspath_from_root(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[tool-policy] train_path={train_path}")
    print(f"[tool-policy] out_dir={out_dir}")

    rows = _read_jsonl(train_path)
    if not rows:
        raise RuntimeError(f"No rows found at {train_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config: Optional[Dict[str, Any]] = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        }

    model_kwargs: Dict[str, Any] = {}
    if quant_config:
        model_kwargs.update(quant_config)
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        **model_kwargs,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    texts: List[str] = []
    for r in rows:
        query = r.get("query") or r.get("question") or ""
        tool_call = r.get("tool_call") or {}
        texts.append(_format_example(tokenizer, query=query, tool_call=tool_call))

    ds = Dataset.from_dict({"text": texts})

    ta_kwargs: Dict[str, Any] = {
        "output_dir": out_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 2,
        "report_to": "none",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
    }
    filtered_ta = _filter_kwargs(TrainingArguments.__init__, ta_kwargs, "TrainingArguments")
    training_args = TrainingArguments(**filtered_ta)

    sft_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": ds,
        "args": training_args,
        "packing": False,
        "dataset_text_field": "text",
        "max_seq_length": args.max_length,
        "max_length": args.max_length,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    filtered_sft = _filter_kwargs(SFTTrainer.__init__, sft_kwargs, "SFTTrainer")

    trainer = SFTTrainer(**filtered_sft)
    trainer.train()

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("[tool-policy] saved to", out_dir)

    # ⏸️ TRAINING CHECKPOINT — Run tool-policy training on Lightning AI before continuing.


if __name__ == "__main__":
    main()
