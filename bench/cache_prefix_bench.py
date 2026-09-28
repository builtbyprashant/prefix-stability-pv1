#!/usr/bin/env python3
"""Cache-prefix benchmark harness for the ReAct-loop caching paper.

Measures how prompt architecture affects prompt-cache effectiveness in an iterative
agent loop, producing the numbers for Table II of `docs/paper_react_cache_prefix.tex`.

Three architectures, identical loop structure (only the prompt assembly differs):

  no_cache : full prompt re-sent every turn, no cache_control.            (baseline)
  phase    : context cached, but the SYSTEM changes on the synthesis turn  (the pitfall)
             -> the largest, last turn cannot read the search turns' cache.
  unified  : one system for the whole loop; context is a cached block that
             grows append-only; per-turn intent trails it.                (the fix)

The loop is driven to a FIXED length N (we don't rely on the model's search/answer
decisions), because we are measuring an input-side caching effect; the model's output
is irrelevant to cache_read / cache_creation. This isolates the architecture variable.

Telemetry comes straight from the provider usage payload:
  input_tokens                 -> fresh, full-price input
  cache_creation_input_tokens  -> written to cache  (billed ~1.25x)
  cache_read_input_tokens      -> read from cache    (billed ~0.1x)
  output_tokens

Safety: DEFAULTS TO DRY-RUN (analytical model, no API calls, no spend). Pass --live to
actually call the API. The API key is read from the ANTHROPIC_ENV (never hardcoded).

Usage:
  python cache_prefix_bench.py                       # dry-run preview (analytical)
  python cache_prefix_bench.py --turns 2,4,6         # more loop lengths
  python cache_prefix_bench.py --live                # real measurement (spends tokens)
  python cache_prefix_bench.py --live --model claude-sonnet-4-6   # 1024-token floor
  python cache_prefix_bench.py --live --out results.csv

Requires: anthropic (already a project dependency).
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
import uuid

# Cache price multipliers relative to base input price (Anthropic): read ~0.1x, write ~1.25x.
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25

# USD per 1M tokens: (input, output).
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

EPHEMERAL = {"type": "ephemeral"}

# Distinct systems so the phase architecture genuinely diverges at the system position.
SYSTEM_UNIFIED = ("You are a research assistant that answers strictly from the CONTEXT in "
                  "the user message. Follow the trailing instruction exactly.")
SYSTEM_SEARCH = ("You are a research assistant deciding whether to search again or answer. "
                 "Reply with exactly one action based on the CONTEXT.")
SYSTEM_SYNTH = ("You are a synthesis module. Produce the final answer directly from the "
                "CONTEXT provided. Be concise and grounded.")

NEXT_ACTION = "Your next action:"
ANSWER_NOW = "You have used all search turns. Give your FINAL answer now from the context above."

# Rough token sizes used ONLY by the analytical dry-run (live mode measures the real thing).
SYS_TOKENS = 40          # all three systems are short and below the cacheable floor
INSTR_TOKENS = 20        # the trailing per-turn instruction


# ----------------------------------------------------------------------------- text gen
_VOCAB = ("the system retrieves relevant context from the indexed documents and reasons over "
          "the evidence to produce a grounded cited answer for the user query about policy").split()


def make_text(approx_tokens: int, seed: str) -> str:
    """Deterministic filler of ~approx_tokens tokens (common words ~1 token each), with the
    seed at the front so each chunk is distinct content (genuine growth, not duplication)."""
    n = max(1, int(approx_tokens))
    return " ".join([seed] + [_VOCAB[i % len(_VOCAB)] for i in range(n)])


# ----------------------------------------------------------------------------- request build
def build_request(arch: str, system_text: str, chunks: list[str], instruction: str):
    """Return (system, messages). The accumulated context is kept as STABLE per-chunk blocks
    (byte-identical across turns); cache_control is placed on the last two context blocks so a
    breakpoint persists at the prior turn's final position. This is what lets a later turn READ
    the earlier prefix: a single growing block with one moving breakpoint never matches, because
    the provider keys cache lookups on breakpoint positions, not on raw token prefixes."""
    if arch == "no_cache":
        ctx = "\n\n".join(chunks)
        return system_text, [{"role": "user", "content": f"CONTEXT:\n{ctx}\n\n{instruction}"}]
    if arch == "sys_cached":
        # The common mistake: cache_control on the (short) SYSTEM prompt; context uncached.
        # The system is far below the cacheable floor, so the provider silently ignores it ->
        # same cost as no_cache, but the developer believes caching is on.
        ctx = "\n\n".join(chunks)
        sysblk = [{"type": "text", "text": system_text, "cache_control": EPHEMERAL}]
        return sysblk, [{"role": "user", "content": f"CONTEXT:\n{ctx}\n\n{instruction}"}]
    blocks = [{"type": "text", "text": (f"CONTEXT:\n{ch}" if i == 0 else ch)}
              for i, ch in enumerate(chunks)]
    for j in {len(blocks) - 1, len(blocks) - 2}:          # <=2 breakpoints: stable prefix + new
        if j >= 0:
            blocks[j]["cache_control"] = EPHEMERAL
    blocks.append({"type": "text", "text": instruction})
    return system_text, [{"role": "user", "content": blocks}]


def system_for(arch: str, k: int, n: int) -> str:
    if arch != "phase":
        return SYSTEM_UNIFIED
    return SYSTEM_SYNTH if k == n else SYSTEM_SEARCH   # the pitfall: system diverges on the last turn


# ----------------------------------------------------------------------------- live call
def call_live(client, model: str, system, messages, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  system=system, messages=messages)
    dt = time.perf_counter() - t0
    u = resp.usage
    return {
        "input": getattr(u, "input_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "latency_s": dt,
    }


# ----------------------------------------------------------------------------- analytical (dry-run)
def usage_analytical(arch: str, k: int, n: int, ctx_tokens: int, prev_ctx_tokens: int,
                     delta_tokens: int) -> dict:
    """Predicted usage from the cost model (Sec. IV of the paper). p_k = SYS + context."""
    p_k = SYS_TOKENS + ctx_tokens
    p_prev = SYS_TOKENS + prev_ctx_tokens
    if arch in ("no_cache", "sys_cached"):
        # sys_cached marks the short system prompt, which is below the floor -> silently
        # ignored -> identical to no_cache.
        return {"input": p_k + INSTR_TOKENS, "cache_read": 0, "cache_creation": 0,
                "output": 16, "latency_s": 0.0}
    diverges = (arch == "phase" and k == n)
    if k == 1 or diverges:
        # nothing readable: write the whole current prefix (a wasted write if it diverges)
        return {"input": INSTR_TOKENS, "cache_read": 0, "cache_creation": p_k,
                "output": 16, "latency_s": 0.0}
    # read the prior prefix, write only the new extension (delta), instruction fresh
    return {"input": INSTR_TOKENS, "cache_read": p_prev, "cache_creation": delta_tokens,
            "output": 16, "latency_s": 0.0}


# ----------------------------------------------------------------------------- loop driver
def run_loop(client, model, arch, n, c0, delta, max_tokens, live, tag="") -> list[dict]:
    # `tag` makes every run's content unique so a cache written by one run can't bleed into
    # a later run within the provider's TTL (which would contaminate the measurement).
    usages, ctx_tokens, prev_ctx_tokens = [], c0, 0
    chunks = [make_text(c0, f"{tag}ctx0")]
    for k in range(1, n + 1):
        system = system_for(arch, k, n)
        instruction = ANSWER_NOW if k == n else NEXT_ACTION
        if live:
            sys_p, msgs = build_request(arch, system, chunks, instruction)
            usages.append(call_live(client, model, sys_p, msgs, max_tokens))
        else:
            usages.append(usage_analytical(arch, k, n, ctx_tokens, prev_ctx_tokens, delta))
        # append-only growth for the next turn (a new stable block)
        prev_ctx_tokens = ctx_tokens
        ctx_tokens += delta
        chunks = chunks + [make_text(delta, f"{tag}ctx{k}")]
    return usages


# ----------------------------------------------------------------------------- metrics
def summarize(usages: list[dict], model: str) -> dict:
    pin, pout = PRICING.get(model, PRICING["claude-haiku-4-5"])
    fresh = sum(u["input"] for u in usages)
    cread = sum(u["cache_read"] for u in usages)
    cwrite = sum(u["cache_creation"] for u in usages)
    out = sum(u["output"] for u in usages)
    billed_in = fresh + cwrite * CACHE_WRITE_MULT + cread * CACHE_READ_MULT
    total_in = fresh + cread + cwrite
    return {
        "fresh_in": fresh, "cache_read": cread, "cache_write": cwrite,
        "hit_rate": (cread / total_in) if total_in else 0.0,
        "billed_in": round(billed_in, 1),
        "cost_usd": round((billed_in * pin + out * pout) / 1e6, 6),
        "latency_s": round(sum(u["latency_s"] for u in usages), 3),
    }


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="claude-haiku-4-5 (2048 floor) | claude-sonnet-4-6 | claude-opus-4-8 (1024 floor)")
    ap.add_argument("--turns", default="2,4", help="comma list of loop lengths N")
    ap.add_argument("--archs", default="no_cache,sys_cached,phase,unified")
    ap.add_argument("--c0-tokens", type=int, default=600, help="initial context size")
    ap.add_argument("--delta-tokens", type=int, default=500, help="context added per turn")
    ap.add_argument("--max-tokens", type=int, default=16, help="output cap (kept small to save cost)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--live", action="store_true", help="actually call the API (spends tokens)")
    ap.add_argument("--out", help="write rows to this CSV")
    args = ap.parse_args()

    turns = [int(x) for x in args.turns.split(",")]
    archs = args.archs.split(",")

    client = None
    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --live needs ANTHROPIC_API_KEY in the environment.", file=sys.stderr)
            return 2
        import anthropic
        client = anthropic.Anthropic()
        n_calls = sum(turns) * len(archs) * args.repeats
        print(f"LIVE: {n_calls} API calls on {args.model} "
              f"(c0={args.c0_tokens}, delta={args.delta_tokens}, max_tokens={args.max_tokens}).")
    else:
        print("DRY-RUN (analytical model, no API calls). Pass --live to measure for real.\n")

    rows = []
    for n in turns:
        for arch in archs:
            samples = []
            for rep in range(args.repeats):
                # A fresh uuid4 per cell makes every run's content globally unique, so a cache
                # entry written by one run (or a prior invocation, within the provider's TTL)
                # can never be read by another. The readable prefix is only for debugging.
                tag = f"{n}-{arch}-{rep}-{uuid.uuid4().hex}-"
                u = run_loop(client, args.model, arch, n, args.c0_tokens, args.delta_tokens,
                             args.max_tokens, args.live, tag=tag)
                samples.append(summarize(u, args.model))
            row = {"N": n, "arch": arch}
            for k in samples[0]:                       # mean + sample std over repeats
                vals = [s[k] for s in samples]
                row[k] = statistics.fmean(vals)
                row[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            rows.append(row)

    # markdown table (means; ± sample std over repeats for cost and latency)
    print(f"\nmodel={args.model}  repeats={args.repeats}  c0={args.c0_tokens}  delta={args.delta_tokens}")
    hdr = ["N", "arch", "hit_rate", "billed_in", "cost_usd", "latency_s"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for r in rows:
        print("| " + " | ".join([
            str(r["N"]), r["arch"],
            f"{r['hit_rate']:.2f}",
            f"{r['billed_in']:.0f}",
            f"{r['cost_usd']:.6f} +/- {r['cost_usd_sd']:.6f}",
            f"{r['latency_s']:.2f} +/- {r['latency_s_sd']:.2f}",
        ]) + " |")

    if args.out:
        keys = ["N", "arch"] + [k for k in rows[0] if k not in ("N", "arch")]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}")

    print("\nRead the per-arch `cache_read`: if it is 0 for every turn, the cache never "
          "fired. The 'phase' row should show writes but little/no read (the pitfall); "
          "'unified' should show growing reads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
