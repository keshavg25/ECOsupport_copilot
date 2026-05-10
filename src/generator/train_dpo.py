import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


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
        print(f"[dpo] {label} dropped keys:", dropped)
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/dpo_pairs.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--sft_adapter",
        default="models/generator_dora",
        help="Path to DoRA/LoRA adapter dir to start DPO from (recommended).",
    )
    ap.add_argument("--out", default="models/generator_dpo")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=0.1)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs_path = _abspath_from_root(args.pairs)
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"DPO pairs file not found: {pairs_path}\n\n"
            "Expected JSONL with keys: prompt, chosen, rejected."
        )

    out_dir = _abspath_from_root(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[dpo] pairs_path={pairs_path}")
    print(f"[dpo] out_dir={out_dir}")

    rows = _read_jsonl(pairs_path)
    if not rows:
        raise RuntimeError(f"No rows found at {pairs_path}")

    # Dataset format expected by TRL DPOTrainer: prompt/chosen/rejected
    ds = Dataset.from_dict(
        {
            "prompt": [r["prompt"] for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
        }
    )

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

    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        **model_kwargs,
    )

    adapter_path = _abspath_from_root(args.sft_adapter)
    if adapter_path and os.path.exists(adapter_path):
        try:
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, adapter_path)
            print(f"[dpo] loaded adapter from {adapter_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from {adapter_path}: {e}\n"
                "If you want to start DPO from the base model only, pass --sft_adapter ''"
            )
    else:
        print("[dpo] no adapter loaded; starting from base model")

    # Reference model: frozen copy of policy weights *before* DPO.
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        **model_kwargs,
    )
    if adapter_path and os.path.exists(adapter_path):
        try:
            from peft import PeftModel

            ref_model = PeftModel.from_pretrained(ref_model, adapter_path)
        except Exception:
            # If adapter load fails for ref but succeeded for policy, we still proceed by using base ref.
            pass

    base_args_kwargs: Dict[str, Any] = {
        "output_dir": out_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        # DPO-specific knobs (some TRL versions store these in DPOConfig instead of trainer kwargs)
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": min(512, args.max_length),
    }

    try:
        from trl import DPOTrainer  # type: ignore
        try:
            from trl import DPOConfig  # type: ignore

            filtered_cfg = _filter_kwargs(DPOConfig.__init__, base_args_kwargs, "DPOConfig")
            training_args = DPOConfig(**filtered_cfg)
            print("[dpo] using TRL DPOConfig")
        except Exception:
            filtered_ta = _filter_kwargs(TrainingArguments.__init__, base_args_kwargs, "TrainingArguments")
            training_args = TrainingArguments(**filtered_ta)
            print("[dpo] using Transformers TrainingArguments")
    except Exception as e:
        raise RuntimeError(
            f"TRL DPOTrainer import failed: {e}. Make sure trl is installed in the environment."
        )

    # Compatibility shim: some TRL versions expect these attributes on args.
    for k, v in {
        "model_init_kwargs": None,
        "ref_model_init_kwargs": None,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": min(512, args.max_length),
    }.items():
        if not hasattr(training_args, k):
            setattr(training_args, k, v)

    dpo_kwargs: Dict[str, Any] = {
        "model": policy,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": ds,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }

    filtered_dpo = _filter_kwargs(DPOTrainer.__init__, dpo_kwargs, "DPOTrainer")
    trainer = DPOTrainer(**filtered_dpo)

    trainer.train()

    # Save final checkpoint
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("[dpo] saved to", out_dir)

    # ⏸️ TRAINING CHECKPOINT — Run DPO on Lightning AI before continuing.


if __name__ == "__main__":
    main()
