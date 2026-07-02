"""Two dedupe passes over validated training pairs:

  1. EVAL-LEAK: drop pairs whose question matches any held-out eval question.
     Prevents the fine-tuned model from looking artificially good on the eval
     because it saw a near-paraphrase during training.

  2. INTRA-DUP: drop pairs whose question matches one already kept earlier
     in the file. Claude generates batches independently, so the same
     question can appear word-for-word multiple times across batches.
     Duplicates inflate effective training size without adding signal.

Two similarity checks per pair:
  - Token Jaccard (set overlap on content tokens, stopwords removed)
  - difflib SequenceMatcher ratio (catches paraphrases / reordering)

Drop if EITHER exceeds the threshold.

Usage:
    uv run python -m dataset.dedupe \\
        --in data/validated/pairs.jsonl \\
        --eval ../energy-text2sql/eval/dataset.jsonl \\
        --out data/validated/pairs_dedup.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "what", "which", "who", "how",
    "many", "much", "did", "do", "does", "return", "show", "list", "find",
    "each", "all", "any", "this", "these", "those", "i", "you", "we",
}

# Eval-leak thresholds — conservative; false positives only cost us a training pair.
LEAK_JACCARD = 0.50
LEAK_SEQRATIO = 0.70

# Intra-dup uses the SQL, not the question. Rationale: in this narrow ERCOT
# domain, similar-sounding questions ("highest 2024" vs "lowest 2024") can
# produce meaningfully different SQL, and dissimilar-sounding ones can
# produce identical SQL. What we want to remove is training redundancy —
# pairs whose SQL teaches the model nothing new beyond what it already saw.
# Drop a pair if its SQL skeleton (literals stripped) matches one we kept.
INTRA_EXACT_SQL = True  # drop on identical normalized SQL
INTRA_SKELETON = True   # also drop on identical SQL skeleton (literals → placeholders)


def normalize(q: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def content_tokens(q: str) -> set[str]:
    """Tokens with stopwords removed, length >= 2."""
    return {t for t in normalize(q).split() if t not in STOPWORDS and len(t) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    s = re.sub(r"\s+", " ", sql.lower()).strip().rstrip(";").strip()
    return s


def sql_skeleton(sql: str) -> str:
    """SQL with all literals replaced by placeholders.

    Two queries that differ only in dates/numbers/strings produce the same
    skeleton, so the model learns the same pattern from both — redundant.
    """
    s = normalize_sql(sql)
    s = re.sub(r"'[^']*'", "'$STR'", s)              # single-quoted strings
    s = re.sub(r'"[^"]*"', '"$STR"', s)              # double-quoted strings
    s = re.sub(r"\b\d+(\.\d+)?\b", "$NUM", s)        # numeric literals
    return s


def is_leak(train_q: str, eval_qs: list[str], eval_tokens: list[set[str]]) -> tuple[bool, str]:
    """Returns (is_leak, reason_with_match)."""
    train_tokens = content_tokens(train_q)
    for eq, et in zip(eval_qs, eval_tokens):
        j = jaccard(train_tokens, et)
        if j >= LEAK_JACCARD:
            return True, f"jaccard={j:.2f} vs eval: {eq[:60]!r}"
        r = seq_ratio(train_q, eq)
        if r >= LEAK_SEQRATIO:
            return True, f"seqratio={r:.2f} vs eval: {eq[:60]!r}"
    return False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True,
                    help="Path to held-out eval JSONL (e.g. energy-text2sql/eval/dataset.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leak-report", type=Path, default=Path("data/raw/leaks.jsonl"))
    ap.add_argument("--intra-dup-report", type=Path, default=Path("data/raw/intra_dups.jsonl"))
    args = ap.parse_args()

    eval_qs = [json.loads(l)["question"] for l in args.eval.open() if l.strip()]
    eval_tokens = [content_tokens(q) for q in eval_qs]
    print(f"Loaded {len(eval_qs)} eval questions to dedupe against\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.leak_report.parent.mkdir(parents=True, exist_ok=True)
    args.intra_dup_report.parent.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: drop eval-leaks
    leaked = 0
    survived_leak = []
    with args.leak_report.open("w") as frep:
        for line in args.inp.open():
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            leak, reason = is_leak(pair["question"], eval_qs, eval_tokens)
            if leak:
                frep.write(json.dumps({**pair, "leak_reason": reason}) + "\n")
                leaked += 1
            else:
                survived_leak.append(pair)
    print(f"Pass 1 (eval-leak): {len(survived_leak)} kept, {leaked} dropped")

    # --- Pass 2: drop intra-dataset duplicates on SQL identity / skeleton
    kept_pairs: list[dict] = []
    seen_sql: dict[str, int] = {}       # normalized SQL -> kept index
    seen_skeleton: dict[str, int] = {}  # SQL skeleton -> kept index
    intra_dropped = 0
    with args.intra_dup_report.open("w") as frep:
        for pair in survived_leak:
            n_sql = normalize_sql(pair["sql"])
            sk = sql_skeleton(pair["sql"])
            dup_of_idx = -1
            reason = ""
            if INTRA_EXACT_SQL and n_sql in seen_sql:
                dup_of_idx = seen_sql[n_sql]
                reason = "exact_sql"
            elif INTRA_SKELETON and sk in seen_skeleton:
                dup_of_idx = seen_skeleton[sk]
                reason = "sql_skeleton"
            if dup_of_idx >= 0:
                frep.write(json.dumps({
                    **pair,
                    "dup_of_question": kept_pairs[dup_of_idx]["question"],
                    "dup_reason": reason,
                }) + "\n")
                intra_dropped += 1
            else:
                seen_sql[n_sql] = len(kept_pairs)
                seen_skeleton[sk] = len(kept_pairs)
                kept_pairs.append(pair)
    print(f"Pass 2 (intra-dup): {len(kept_pairs)} kept, {intra_dropped} dropped")

    # --- Write final output
    with args.out.open("w") as fout:
        for pair in kept_pairs:
            fout.write(json.dumps(pair) + "\n")

    print(f"\nFinal: {len(kept_pairs)} unique pairs → {args.out}")
    if intra_dropped:
        print("\nSample intra-dups:")
        with args.intra_dup_report.open() as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                d = json.loads(line)
                print(f"  - {d['question'][:80]!r}")
                print(f"    dup of: {d['dup_of_question'][:80]!r}")
    print(f"\nNext: uv run python -m dataset.format --in {args.out} --out-dir data/sft/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
