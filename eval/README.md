# Eval integration

This project doesn't ship its own eval harness — it **reuses** the 12-question
result-equivalence harness from `energy-text2sql`. That keeps the comparison
fair (same questions, same gold SQL, same scorer) and avoids duplicated code.

## How to wire it in

After training and downloading the adapter:

```bash
# 1. Copy this project's inference wrapper into energy-text2sql
cp inference/finetuned_agent.py ../energy-text2sql/agent/finetuned.py

# 2. In energy-text2sql/eval/run.py, add a --agent finetuned branch:
#       case "finetuned": from agent.finetuned import FinetunedAgent; ...

# 3. Run the eval there
cd ../energy-text2sql
uv run python eval/run.py --agent finetuned --save
```

The output JSON lands in `energy-text2sql/eval/runs/`. Compare against
`baseline`, `multi`, and `multi+RAG` runs from the same harness.

## What "correct" means

The energy-text2sql scorer compares **result rows**, not SQL strings:
sort-insensitive, float-tolerant. See `eval/scorer.py` in that repo and the
`eval/README.md` for known limitations (column-count strictness, summary
hallucinations not detected, etc.).

## Expected outcome

A 1.5B fine-tuned model on a few hundred Claude-generated training pairs
is very unlikely to hit 12/12. Realistic range: 5-9/12. The point of the
project is **the workflow**, not the result.

If the fine-tuned model materially beats the base + prompting (e.g.
8/12 vs 3/12), that's the story: domain SFT helps a small model close
some of the gap to a frontier API on a narrow task.
