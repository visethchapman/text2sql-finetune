# Lessons

Notes from building this project. Kept separate from README so they can be
folded in after training results land.

## 1. Claude-generated training data is 35% redundant

Ran 500 pairs through 20 independent Claude batches. After dedupe:

| Stage | Kept | Dropped |
|---|---|---|
| Raw | 500 | — |
| Executes on Postgres | 497 | 3 |
| Not an eval-set leak | 475 | 22 |
| **Unique SQL skeleton** | **310** | **165** |

Cost: ~$0.89 for 500 raw → 310 unique → **~$0.003 per usable pair**.

**Root cause:** independent batches + narrow domain = Claude keeps producing
the same handful of natural questions.

**Fix if you cared to:** stratify each batch by SQL pattern, and pass prior
skeletons back as an exclusion list. Realistic lift: 35% → 15% redundancy.
Not worth retrofitting; dedupe post-hoc is cheaper than prompt engineering.

**Rule of thumb:** dedupe first, prompt-engineer for diversity only if
post-hoc dedupe leaves you short.

---

## 2. GPU compatibility hell on Kaggle free tier

Started with QLoRA (4-bit). Kaggle's free-tier GPUs both had issues:

| Attempt | GPU | Result |
|---|---|---|
| 1 | T4 x2, QLoRA | Crashed — cross-entropy tensors split across `cuda:0` and `cuda:1` |
| 2 | P100, QLoRA | Kernel died — P100 is sm_60; bitsandbytes needs sm_70+ |
| 3 | P100, plain LoRA (fp16) | Kernel died — **PyTorch itself dropped sm_60 kernels** (needs sm_70+) |
| 4 | T4 x2, LoRA, `device_map={'': 0}` | Still crashed — Accelerate still saw 2 GPUs and split the batch across them |
| 5 | T4 x2, LoRA, `CUDA_VISIBLE_DEVICES=0` set before torch imports | Worked |

Three independent traps:
- **P100 is a hardware dead end** for modern PyTorch/transformers. Current PyTorch ships kernels for sm_70+ only. If Kaggle offers you P100, walk away — it's for legacy TensorFlow / older PyTorch only.
- **T4 x2 with `device_map='auto'`** silently splits the model across both cards, then loss compute crashes on mixed devices.
- **`device_map={'': 0}` alone isn't enough** — Accelerate still autodetects GPU 1 and dispatches inputs there. On a multi-GPU node where you only want one, set `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` *before* the first `import torch`. This hides GPU 1 from the process entirely.

## 3. QLoRA wasn't earning its keep anyway

QLoRA exists to fit 7B+ models into small VRAM by 4-bit quantizing the base. A **1.5B** model in fp16 is only ~3 GB — fits any 16 GB card with headroom to spare.

**Rule of thumb:**
- ≤ 3B model on 16 GB → plain LoRA in fp16. Simpler, portable across GPUs.
- 7B+ model on 16 GB → QLoRA in 4-bit. The compression is what buys you the fit.
- Pre-Volta GPU (sm_60 and older) → find a different GPU. Modern PyTorch won't run.

Use QLoRA when you need the compression, not because the acronym is fancier.
