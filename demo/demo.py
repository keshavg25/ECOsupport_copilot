import argparse
import json

from src.pipeline.copilot import EcoSupportCopilot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)

    ap.add_argument("--tool_policy_base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tool_policy_adapter", default="models/tool_policy")

    ap.add_argument("--generator_base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--generator_adapter", default="models/generator_dpo")

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    copilot = EcoSupportCopilot(
        tool_policy_base=args.tool_policy_base,
        tool_policy_adapter=args.tool_policy_adapter,
        generator_base=args.generator_base,
        generator_adapter=args.generator_adapter,
        top_k_default=args.top_k,
    ).load()

    answer, trace = copilot.answer(args.question, top_k=args.top_k, max_new_tokens=args.max_new_tokens)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== TOOL_TRACE_JSON ===\n")
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
