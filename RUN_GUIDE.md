# EcoSupport-Copilot — Run Guide

This repo contains a retrieval+rera​nking pipeline plus a tool-policy + ReAct-style loop.

## Demo (CLI)

```bash
python -u demo/demo.py \
  --question "What is your return policy for unopened items?" \
  --tool_policy_adapter models/tool_policy \
  --generator_adapter models/generator_dpo
```

Output includes `=== ANSWER ===` and `=== TOOL_TRACE_JSON ===`.

## Evaluation

Runs a small default set of questions and writes a CSV with basic metrics.

```bash
python -u src/evaluation/run_eval.py \
  --tool_policy_adapter models/tool_policy \
  --generator_adapter models/generator_dpo \
  --out_csv artifacts/eval_results.csv
```

Optionally provide your own questions JSONL:

```jsonl
{"id":"q1","question":"..."}
{"id":"q2","question":"..."}
```

```bash
python -u src/evaluation/run_eval.py --questions_jsonl data/eval/questions.jsonl
```

## Serving (FastAPI)

```bash
# Windows PowerShell
$env:TOOL_POLICY_ADAPTER="models/tool_policy"
$env:GENERATOR_ADAPTER="models/generator_dpo"
uvicorn src.pipeline.serving:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is your return policy for unopened items?","top_k":5,"max_new_tokens":220}'
```
