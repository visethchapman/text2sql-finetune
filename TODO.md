# TODO

Tracked work and intentional skips for `text2sql-finetune`.

## v1 scope (this build)

- [x] Repo scaffold
- [ ] Day 1 — dataset generator (Claude → ~1000 Q/SQL pairs)
- [ ] Day 2 — validate (drop SQL that doesn't execute) + dedupe vs eval set + format (chat template)
- [ ] Day 3 — Kaggle QLoRA training notebook
- [ ] Day 4 — inference wrapper + eval harness integration
- [ ] Day 5 — full eval run + push adapter to HF Hub
- [ ] Day 6 — iterate hyperparams
- [ ] Day 7 — README polish + headline plot + HF model card

## Out of v1 scope (deliberate)

### DPO post-training
- Would need preference pairs (winner / loser SQL for the same question).
- Could generate from eval failures, but adds 2-3 days of work for marginal portfolio gain.
- Revisit as v2 once v1 ships.

### Continued pre-training on raw SQL corpus
- Honest "pre-train" experience would require domain-adaptive pre-training on a
  large unlabeled SQL corpus before SFT. Out of scope for a 1.5B portfolio build
  on a tiny domain.

### Bigger models (7B+)
- Doesn't fit Kaggle T4 even with QLoRA. Needs paid Modal/RunPod time.
- Marginal gains over 1.5B-Coder on this small a domain; not worth the cost for v1.

### Real preference data via human / LLM-judge ratings
- Eval-failures-as-DPO-pairs is the cheap path; real preference collection is
  out of scope.

## Known limitations to document in README

- Synthetic Claude-generated training data inherits Claude's quirks.
- 12-question eval is too small for statistical confidence; conclusions
  directional only.
- LoRA rank/alpha/lr not searched — using sensible defaults from TRL examples.

## Decisions locked in

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (already SQL-aware, fits T4)
- Method: QLoRA, rank 16, alpha 32, lr 2e-4, 3 epochs
- GPU: Kaggle T4 free tier (no payment friction)
- Eval reuses `energy-text2sql/eval/run.py` — no duplicated harness
