"""Generate synthetic question/SQL training pairs against the ERCOT schema.

Day 1 deliverable. Reads schema metadata from the running Postgres
container in `energy-text2sql`, prompts Claude N times, writes JSONL.

Validation (drop-failures) happens in `dataset/validate.py`.

Usage:
    uv run python -m dataset.generate --n 1000 --out data/raw/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


SCHEMA_DUMP = """
-- ERCOT-only electricity demand + Houston weather, loaded by energy-text2sql/etl

eia.demand (
    region       TEXT             NOT NULL,    -- BA code; only ERCO loaded
    period       TIMESTAMPTZ      NOT NULL,    -- hour in UTC
    value        DOUBLE PRECISION,             -- demand in MWh
    value_units  TEXT,
    PRIMARY KEY (region, period)
)

noaa.stations (
    station_id           TEXT PRIMARY KEY,
    name                 TEXT,
    state                TEXT,
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    elevation_m          DOUBLE PRECISION,
    nearest_eia_region   TEXT
)

noaa.daily_weather (
    station_id  TEXT             NOT NULL REFERENCES noaa.stations(station_id),
    obs_date    DATE             NOT NULL,
    tmax_c      DOUBLE PRECISION,            -- daily max temp, deg C
    tmin_c      DOUBLE PRECISION,            -- daily min temp, deg C
    prcp_mm     DOUBLE PRECISION,            -- precipitation, mm
    awnd_ms     DOUBLE PRECISION,            -- average wind speed, m/s
    PRIMARY KEY (station_id, obs_date)
)

-- All demand timestamps are UTC. Houston weather is local-date.
-- For cross-domain joins, cast: (period AT TIME ZONE 'America/Chicago')::date
-- Data range: 2020-01-01 to 2024-12-31
""".strip()


PROMPT = f"""You are generating training data for a SQL code model. Given the
ERCOT electricity-demand + Houston weather Postgres schema below, produce
a batch of {{batch_size}} diverse natural-language question / SQL pairs.

## Schema

{SCHEMA_DUMP}

## Requirements

1. Questions cover a mix: peaks/minimums, aggregations (daily/monthly/yearly),
   temperature correlations, specific events (Feb 2021 freeze, summer peaks),
   window functions, CTEs, simple lookups.
2. SQL must be valid Postgres, must execute against the schema above.
3. Use `(period AT TIME ZONE 'America/Chicago')::date` ONLY when joining
   demand with weather (UTC->local date alignment). Pure demand queries
   stay in UTC.
4. Return ONLY columns the question asks about — no extras.
5. Vary difficulty: ~30% easy (single table, no aggregation), ~50% medium
   (single aggregation or join), ~20% hard (window funcs, multi-CTE).
6. Each question must be unique within the batch.

## Output

Respond with a JSON array, no prose. Each item:
{{{{
    "question": "...",
    "sql": "..."
}}}}
"""


def fetch_table_counts(conn: psycopg.Connection) -> dict[str, int]:
    """Quick sanity check that the DB is loaded."""
    counts = {}
    for tbl in ("eia.demand", "noaa.daily_weather", "noaa.stations"):
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            counts[tbl] = cur.fetchone()[0]
    return counts


def generate_batch(client: Anthropic, batch_size: int) -> list[dict]:
    """One Claude call → batch_size pairs. Retries once on JSON parse failure."""
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT.format(batch_size=batch_size)}],
    )
    text = msg.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Target total pairs")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="Pairs per Claude call. Larger = fewer calls but more JSON parse risk.")
    ap.add_argument("--out", type=Path, default=Path("data/raw/pairs.jsonl"))
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    db_url = os.environ.get("DB_URL", "postgresql://energy:energy_dev@localhost:5432/energy")

    with psycopg.connect(db_url) as conn:
        counts = fetch_table_counts(conn)
    print(f"DB sanity: {counts}")
    if counts["eia.demand"] == 0:
        print("ERROR: eia.demand is empty — run energy-text2sql ETL first", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = Anthropic(api_key=api_key)

    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    total = 0
    with args.out.open("w") as f:
        for i in range(n_batches):
            try:
                pairs = generate_batch(client, args.batch_size)
            except Exception as e:
                print(f"  batch {i+1}/{n_batches}: FAILED ({e}); skipping", file=sys.stderr)
                continue
            for p in pairs:
                if "question" in p and "sql" in p:
                    f.write(json.dumps(p) + "\n")
                    total += 1
            print(f"  batch {i+1}/{n_batches}: +{len(pairs)} (total {total})")

    print(f"\nWrote {total} pairs to {args.out}")
    print(f"Next: uv run python -m dataset.validate --in {args.out} --out data/validated/pairs.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
