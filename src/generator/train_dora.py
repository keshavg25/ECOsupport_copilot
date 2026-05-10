import argparse
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


def _guess_target_modules(model_name: str) -> List[str]:
    name = model_name.lower()
    # Works for Llama/Qwen/Mistral-style MLP+attn projections.
    if any(k in name for k in ["llama", "qwen", "mistral", "gemma"]):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    # Fallback: still a reasonable default.
    return ["q_proj", "v_proj"]


def _format_example(
    tokenizer: Any,
    system: str,
    user: str,
    context: str,
    assistant: str,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Question:\n{user}\n\nRetrieved Evidence:\n{context}",
        },
        {"role": "assistant", "content": assistant},
    ]

    # Prefer chat template when available.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Generic fallback that works across base CausalLMs.
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\nQuestion:\n{user}\n\nRetrieved Evidence:\n{context}\n"
        f"<|assistant|>\n{assistant}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/generator_train.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="models/generator_dora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--attn_implementation", default=None)
    args = ap.parse_args()

    train_path = _abspath_from_root(args.train)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n\n"
            "Expected a JSONL with keys: system,user,context,assistant."
        )

    out_dir = _abspath_from_root(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[dora] train_path={train_path}")
    print(f"[dora] out_dir={out_dir}")

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

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        **model_kwargs,
    )

    target_modules = _guess_target_modules(args.base_model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_dora=True,
    )
    model = get_peft_model(model, peft_config)

    texts: List[str] = []
    for r in rows:
        texts.append(
            _format_example(
                tokenizer=tokenizer,
                system=r.get("system", ""),
                user=r.get("user", ""),
                context=r.get("context", ""),
                assistant=r.get("assistant", ""),
            )
        )

    ds = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy="no" if args.eval_steps <= 0 else "steps",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        report_to="none",
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        args=training_args,
        packing=False,
    )

    trainer.train()

    # Save final adapter + tokenizer
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("[dora] saved to", out_dir)

    # ⏸️ TRAINING CHECKPOINT — DoRA generator training is a heavier run.
    # Run this on Lightning AI before continuing to DPO.


if __name__ == "__main__":
    main()
