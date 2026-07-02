"""Execute each (question, sql) pair against Postgres. Keep only those that
return >=1 row without error. Drops syntactically-bad SQL, hallucinated
columns, and queries that return empty (likely wrong-date / wrong-filter).

Usage:
    uv run python -m dataset.validate --in data/raw/pairs.jsonl --out data/validated/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


def validate_pair(conn: psycopg.Connection, sql: str, timeout_ms: int = 5000) -> tuple[bool, str]:
    """Run SQL in a savepoint; rollback regardless. Returns (ok, reason)."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
            cur.execute(sql)
            rows = cur.fetchall()
            if not rows:
                return False, "empty_result"
            return True, "ok"
    except Exception as e:
        return False, f"exec_error: {type(e).__name__}: {str(e)[:120]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reject-log", type=Path, default=Path("data/raw/rejected.jsonl"),
                    help="Where to write rejected pairs + reasons (for debugging)")
    args = ap.parse_args()

    db_url = os.environ.get("DB_URL", "postgresql://energy:energy_dev@localhost:5432/energy")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.reject_log.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    reasons: dict[str, int] = {}

    with psycopg.connect(db_url, autocommit=False) as conn, \
            args.inp.open() as fin, \
            args.out.open("w") as fout, \
            args.reject_log.open("w") as freject:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            ok, reason = validate_pair(conn, pair["sql"])
            conn.rollback()  # discard whatever the SQL touched
            if ok:
                fout.write(json.dumps(pair) + "\n")
                kept += 1
            else:
                freject.write(json.dumps({**pair, "reject_reason": reason}) + "\n")
                rejected += 1
                key = reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            if line_num % 100 == 0:
                print(f"  {line_num}: kept={kept}, rejected={rejected}")

    print(f"\nValidated: {kept} kept, {rejected} rejected")
    print(f"Rejection breakdown:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nNext: uv run python -m dataset.format --in {args.out} --out-dir data/sft/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
