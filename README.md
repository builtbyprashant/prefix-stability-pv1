# Prefix Stability for Prompt Caching in LLM Agents — reproduction repo

Frozen code, data, and derivations for the paper *Prefix Stability for Prompt Caching in LLM
Agents: When It Helps and When It Backfires* (GlobalSouthAI @ NeurIPS 2026). This repo is the
**single source of truth**: every number quoted in the camera-ready regenerates from the frozen
data here.

## Reproduce every paper number

```bash
python reproduce_paper_numbers.py
```

Reads the frozen data + the analytical model, recomputes each quoted number, and asserts it
matches the printed camera-ready. Exit code 0 iff all match (currently **46/46**). See
[`PAPER_NUMBERS.md`](PAPER_NUMBERS.md) for a number-by-number provenance map, including the
handful of inputs (provider constants and external citations) that are recorded rather than
reproduced.

No API key or network needed to reproduce — everything runs off the committed data.

## Layout

```
reproduce_paper_numbers.py   # regenerates + asserts every paper number (start here)
PAPER_NUMBERS.md             # each paper number -> its source and formula
cost_model.py                # analytical cost model (worked example)
bench/
  cache_prefix_bench.py      # cost / hit-rate measurement harness (4 architectures)
  cost_haiku.csv             # Table 2 (Haiku, c0~5000)      -- measured, sigma=0
  cost_sonnet.csv            # Sonnet comparison (smaller ctx) -- measured, sigma=0
  floor_haiku.csv            # cacheable-floor demo (Haiku caches nothing at c0~1500)
ab_experiment.py             # answer-quality A/B harness (H4)
tasks.json                   # 36 QA items (24 baseline + 12 hard, id prefix `h_`)
results/
  raw_full.jsonl             # every answer + judgement (H4)  -- frozen
  summary_full.json          # H4 means, diff, CI, pairwise   -- frozen
  raw_smoke.jsonl / summary_smoke.json
```

## The three number families

1. **Analytical cost model** (`cost_model.py`) — the worked example (1660/1810/1370, 4920/3830/2670)
   from the paper's formulas.
2. **Cost / hit-rate** (`bench/`) — Table 2 (Haiku), the Sonnet comparison, and the floor claims.
   Token counts are deterministic (σ=0), so the CSVs reproduce exactly. Harness defaults to a
   no-spend analytical dry-run; `--live` measures against the API.
3. **Answer quality, H4** (`ab_experiment.py`, `results/`, `tasks.json`) — Table 3, the pairwise
   ties, and the non-inferiority CI. The generation/judging is live and non-deterministic
   (temperature 0.7, no seed parameter), so `results/raw_full.jsonl` is the frozen record; the
   paper numbers are derived from it (the bootstrap CI is seeded, so it recomputes exactly).

### H4 design (answer-quality A/B)

Two complete 2-turn loops (explore → synthesize) that differ in **exactly one thing**: where the
per-turn instructions live. The instruction *text* is byte-identical across arms.

| | System prompt | Per-turn intent |
|---|---|---|
| **Arm C** (phase-switched) | changes per phase (`ROLE+EXPLORE`, then `ROLE+SYNTH`) | in the system prompt |
| **Arm D** (unified) | invariant (`ROLE`) across the loop | in the user message |

- **Generator:** `claude-haiku-4-5`, temperature 0.7. **Judge:** `claude-sonnet-4-6` (a different,
  stronger model, to avoid self-preference), pointwise correctness + a blind order-randomized
  pairwise winner.
- **Metric:** non-inferiority, not superiority. Paired difference `mean(D) − mean(C)` with a 95%
  paired bootstrap CI over tasks; margin `delta = 0.05`; parity holds if the CI lower bound is
  above `−delta`.

## Regenerating the raw data (optional; costs API tokens)

The committed data is the source of truth. To re-measure, see the commands at the bottom of
[`PAPER_NUMBERS.md`](PAPER_NUMBERS.md). The API key is read from `ANTHROPIC_API_KEY` in the
environment; it is never hardcoded.

## Note

Kept private and separate from the public `genai-doc-assistant` repository; the task corpus is
three public encyclopedia articles (AI, AGI, ML). This is a per-submission reproduction snapshot,
frozen at the accepted-paper state.
