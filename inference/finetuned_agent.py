"""Local inference wrapper for the fine-tuned model. Implements the
same Agent protocol as energy-text2sql/agent/base.py so eval/run.py can
treat it identically.

After training on Kaggle and pushing the adapter to HF Hub, run this
locally:

    uv run python -m inference.finetuned_agent --adapter visethchapman/ercot-text2sql-qwen-1.5b-lora

To plug into the energy-text2sql eval harness, copy this file into that
repo's agent/ directory and add a `--agent finetuned` branch to eval/run.py.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from training.config import ADAPTER_REPO, BASE_MODEL


# Mirror of energy-text2sql/agent/base.py::AgentResult — kept inline so this
# file is self-contained when copied into the other repo.
@dataclass
class AgentResult:
    sql: str | None = None
    result_rows: list[tuple] | None = None
    result_columns: list[str] | None = None
    final_answer: str | None = None
    error: str | None = None
    category: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_sec: float = 0.0
    extra: dict = field(default_factory=dict)


# Loaded lazily on first call — torch + transformers are heavy imports.
_MODEL = None
_TOKENIZER = None


def _load(adapter_repo: str = ADAPTER_REPO, base_model: str = BASE_MODEL):
    """Load base + LoRA adapter once. Uses bf16 on Mac MPS, fp16 on CUDA, fp32 on CPU."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    _TOKENIZER = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    _MODEL = PeftModel.from_pretrained(base, adapter_repo).to(device)
    _MODEL.eval()
    return _MODEL, _TOKENIZER


SYSTEM_PROMPT = """You are a Postgres SQL expert for ERCOT electricity-demand and Houston weather data.

Schema:
eia.demand(region, period, value, value_units) — region='ERCO'; period in UTC; value in MWh
noaa.daily_weather(station_id, obs_date, tmax_c, tmin_c, prcp_mm, awnd_ms) — Houston; obs_date is local date
noaa.stations(station_id, name, state, nearest_eia_region)

Notes: All demand is UTC. Houston weather is local date. For joins,
cast period to local date: (period AT TIME ZONE 'America/Chicago')::date

Return ONLY valid Postgres SQL. No explanation, no markdown fences."""


class FinetunedAgent:
    name = "finetuned"

    def __init__(self, adapter_repo: str = ADAPTER_REPO):
        self.adapter_repo = adapter_repo

    def answer(self, question: str, conn: psycopg.Connection) -> AgentResult:
        import torch
        model, tok = _load(self.adapter_repo)
        t0 = time.time()

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        inp = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        inp = inp.to(model.device)
        in_tokens = inp.shape[1]

        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
        sql = tok.decode(out[0][in_tokens:], skip_special_tokens=True).strip()
        out_tokens = out.shape[1] - in_tokens

        # Execute against Postgres
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description] if cur.description else []
            conn.commit()
            return AgentResult(
                sql=sql, result_rows=rows, result_columns=cols,
                input_tokens=in_tokens, output_tokens=out_tokens,
                latency_sec=time.time() - t0,
            )
        except Exception as e:
            conn.rollback()
            return AgentResult(
                sql=sql, error=str(e)[:200], category="sql_error",
                input_tokens=in_tokens, output_tokens=out_tokens,
                latency_sec=time.time() - t0,
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER_REPO)
    ap.add_argument("--question", default="What was peak hourly ERCOT demand in 2024?")
    ap.add_argument("--db", default="postgresql://energy:energy_dev@localhost:5432/energy")
    args = ap.parse_args()

    agent = FinetunedAgent(adapter_repo=args.adapter)
    with psycopg.connect(args.db) as conn:
        res = agent.answer(args.question, conn)

    print(f"\n--- SQL ---\n{res.sql}")
    if res.error:
        print(f"\nERROR: {res.error}")
    elif res.result_rows:
        print(f"\n--- Result ({len(res.result_rows)} rows) ---")
        print(res.result_columns)
        for r in res.result_rows[:10]:
            print(r)
    print(f"\nlatency: {res.latency_sec:.2f}s | in/out tokens: {res.input_tokens}/{res.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
