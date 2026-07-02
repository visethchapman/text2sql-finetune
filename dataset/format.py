"""Convert validated (question, sql) pairs into Qwen chat-template JSONL
for SFTTrainer. Splits 90/5/5 train/val/test.

The system prompt + schema is identical for every example; the user turn
is the question, the assistant turn is the SQL.

Usage:
    uv run python -m dataset.format --in data/validated/pairs.jsonl --out-dir data/sft/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Same schema dump used in dataset/generate.py — keep these in sync.
SCHEMA = """eia.demand(region, period, value, value_units) — region='ERCO'; period in UTC; value in MWh
noaa.daily_weather(station_id, obs_date, tmax_c, tmin_c, prcp_mm, awnd_ms) — Houston station; obs_date is local date
noaa.stations(station_id, name, state, nearest_eia_region)

Notes: All demand is UTC. Houston weather is local date. For joins,
cast period to local date: (period AT TIME ZONE 'America/Chicago')::date"""

SYSTEM = f"""You are a Postgres SQL expert for ERCOT electricity-demand and Houston weather data.

Schema:
{SCHEMA}

Return ONLY valid Postgres SQL. No explanation, no markdown fences."""


def to_chat(pair: dict) -> dict:
    """Qwen chat template — three messages, no tools."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["sql"]},
        ]
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.inp.open() as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} validated pairs")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_val = max(1, n // 20)
    n_test = max(1, n // 20)
    n_train = n - n_val - n_test

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for name, subset in splits.items():
        out = args.out_dir / f"{name}.jsonl"
        with out.open("w") as f:
            for p in subset:
                f.write(json.dumps(to_chat(p)) + "\n")
        print(f"  {name}: {len(subset):>5} → {out}")

    print(f"\nNext: zip {args.out_dir}/*.jsonl and upload to Kaggle as a Dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
