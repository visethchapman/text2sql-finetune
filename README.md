# text2sql-finetune

LoRA fine-tune of a small open LLM (Qwen2.5-Coder-1.5B) on ERCOT text-to-SQL.
Companion to [`energy-text2sql`](https://github.com/visethchapman/energy-text2sql) —
same Postgres schema, same eval harness, different question:

> **Can a 1.5B specialist match a frontier general model on a narrow domain task?**

Not "can I beat Claude" — a 1.5B model with a few hundred training pairs almost
certainly won't. The interesting question is *how close* it gets, and whether
the workflow (data gen → SFT → eval) is sound.

> **Status:** scaffold complete, dataset + training pending.

---

## Plan

| Day | Deliverable |
|---|---|
| 1 | Repo scaffold + `dataset/generate.py` (Claude → ~1000 Q/SQL pairs against ERCOT schema) |
| 2 | `dataset/validate.py` — execute on Postgres, drop failures; `dataset/format.py` — chat-template JSONL, train/val/test split |
| 3 | Kaggle notebook with QLoRA + `trl.SFTTrainer`; first training run on T4 |
| 4 | `inference/finetuned_agent.py` — load base + adapter, plug into `energy-text2sql` eval harness |
| 5 | Run eval (fine-tuned vs base + prompting vs Claude); push adapter to HF Hub |
| 6 | Iterate hyperparams; second training run if needed |
| 7 | README, headline plot, model card |

---

## Stack

| Layer | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Method | QLoRA (4-bit base + LoRA adapter) |
| Training | HuggingFace `trl.SFTTrainer` + `peft` |
| GPU | Kaggle T4 (free, 30 hr/week) |
| Eval | Reuses `energy-text2sql/eval/run.py` 12-question harness |

---

## Local workflow

```bash
# 1. dataset generation (requires Postgres from energy-text2sql)
cd ../energy-text2sql && docker compose up -d && cd -
cp .env.example .env  # paste keys
uv sync

uv run python -m dataset.generate --n 500 --out data/raw/pairs.jsonl
uv run python -m dataset.validate --in data/raw/pairs.jsonl --out data/validated/pairs.jsonl
uv run python -m dataset.dedupe --in data/validated/pairs.jsonl \
    --eval ../energy-text2sql/eval/dataset.jsonl \
    --out data/validated/pairs_dedup.jsonl
uv run python -m dataset.format --in data/validated/pairs_dedup.jsonl --out-dir data/sft/
# upload data/sft/ to Kaggle as a Dataset

# 2. training happens on Kaggle (see training/train_kaggle.ipynb)

# 3. eval (after downloading adapter from HF Hub)
uv run python -m inference.finetuned_agent --adapter visethchapman/ercot-text2sql-qwen-1.5b-lora
```

---

## Honest caveats

- **Synthetic training data from Claude inherits Claude's biases.** Documented up front.
- **Eval set is 12 questions** — small. Conclusions are directional, not statistically tight.
- **A 1.5B model will not beat a frontier API.** Story is the workflow, not the win.

See [TODO.md](TODO.md) for what's intentionally out of scope.

---

## License

MIT — see [LICENSE](LICENSE).
