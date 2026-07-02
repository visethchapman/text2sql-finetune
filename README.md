# text2sql-finetune

LoRA fine-tune of [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
on ERCOT text-to-SQL. Trains on Kaggle's free T4 in ~5 min, runs locally on
a Mac at $0/query.

Companion to [`energy-text2sql`](https://github.com/visethchapman/energy-text2sql):
same Postgres schema, same 12-question eval harness.

> **Adapter on HuggingFace Hub:** [visethchapman/ercot-text2sql-qwen-1.5b-lora](https://huggingface.co/visethchapman/ercot-text2sql-qwen-1.5b-lora)

---

## Scoreboard

| Model | Correct | Cost / query | Avg latency |
|---|---|---|---|
| Qwen2.5-Coder-1.5B raw (no LoRA) | 2/12 (17%) | $0 | 5.3s |
| **Qwen2.5-Coder-1.5B + our LoRA** | **6/12 (50%)** | **$0** | **3.1s** |
| Claude Sonnet 4.5 (single-call) | 12/12 (100%) | ~$0.05 | 4.6s |
| Claude Sonnet 4.5 multi-agent | 12/12 (100%) | ~$0.10 | 9.6s |

Fine-tuning **tripled the base score** (2 → 6) at zero incremental inference
cost. The model handles simple aggregations (peaks, totals, summer averages)
but still fails on advanced SQL rules (GROUP BY with non-aggregated columns,
alias-in-ORDER-BY) and cross-domain joins with timezone casts.

---

## What one training pair looks like

Each pair is a natural-language question paired with the SQL that answers it.
The training data is 280 such pairs in HuggingFace chat-template format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a Postgres SQL expert for ERCOT... Schema: eia.demand(region, period, value...)"},
    {"role": "user",   "content": "What was peak hourly ERCOT demand in 2024?"},
    {"role": "assistant", "content": "SELECT MAX(value) FROM eia.demand WHERE region='ERCO' AND EXTRACT(YEAR FROM period)=2024;"}
  ]
}
```

The model learns to map `(schema + question) → SQL`. Not `question → SQL` —
that's why the schema must be in the system prompt at inference too.

---

## Stack

| Layer | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Method | Plain LoRA in fp16 (see [LESSONS.md](LESSONS.md) for why not QLoRA) |
| Training | HuggingFace `trl.SFTTrainer` + `peft.LoraConfig` |
| Trainable params | 18.5M / 1.56B (1.18%) |
| GPU | Kaggle T4 (free tier, ~4.5 min wall clock) |
| Data generation | Claude Sonnet 4.5 API — 500 raw pairs |
| Data validation | Executed each SQL on real Postgres |
| Deploy | HuggingFace Hub |
| Local inference | `transformers` + `peft` on Mac MPS |

---

## Dataset pipeline

```
500 raw pairs from Claude
  ↓ validate: execute SQL on Postgres, drop failures      (–3)
497
  ↓ eval-leak dedupe: fuzzy match vs 12 held-out eval qs  (–22)
475
  ↓ intra dedupe: same SQL skeleton (literals stripped)   (–165)
310 unique  →  280 train  /  15 val  /  15 test
```

35% intra-redundancy is a real finding — see [LESSONS.md](LESSONS.md#1-claude-generated-training-data-is-35-redundant).

---

## Reproduce it

```bash
# 1. dataset generation — requires Postgres from energy-text2sql
cd ../energy-text2sql && docker compose up -d && cd -
cp .env.example .env  # paste ANTHROPIC_API_KEY, HF_TOKEN
uv sync

uv run python -m dataset.generate --n 500 --out data/raw/pairs.jsonl
uv run python -m dataset.validate --in data/raw/pairs.jsonl --out data/validated/pairs.jsonl
uv run python -m dataset.dedupe --in data/validated/pairs.jsonl \
    --eval ../energy-text2sql/eval/dataset.jsonl \
    --out data/validated/pairs_dedup.jsonl
uv run python -m dataset.format --in data/validated/pairs_dedup.jsonl --out-dir data/sft/

# 2. training on Kaggle — see training/train_kaggle.ipynb

# 3. eval locally, wired into energy-text2sql
cd ../energy-text2sql
uv run python eval/run.py --agent qwen_base --save   # baseline (no LoRA)
uv run python eval/run.py --agent finetuned --save   # with our LoRA
```

---

## Cost tally

| Item | Cost |
|---|---|
| Claude API for 500 training pairs | $0.89 |
| Kaggle T4 GPU (free tier) | $0 |
| HuggingFace Hub hosting | $0 |
| Local inference (per query) | $0 (electricity only) |
| **Total** | **~$0.89** |

---

## Honest caveats

- **Synthetic training data from Claude inherits Claude's biases.**
- **12-question eval is small** — conclusions are directional.
- **A 1.5B model does not beat frontier APIs.** Story is proximity, not victory.

---

## License

MIT — see [LICENSE](LICENSE). Base model retains its own Qwen2.5 license.
