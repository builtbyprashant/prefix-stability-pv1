# Where every camera-ready number comes from

This repo is the single source of truth for the paper *Prefix Stability for Prompt Caching in
LLM Agents: When It Helps and When It Backfires*. One command regenerates every number quoted
in the camera-ready from the frozen data here and asserts each matches the printed page:

```bash
python reproduce_paper_numbers.py     # exit 0 iff all reproducible numbers match
```

At the last run: **46/46 reproducible numbers match.**

## Result numbers (reproduced from this repo)

| Paper location | Number(s) | Source in this repo | How |
|---|---|---|---|
| Cost Model, worked example | N=2 1660 / 1810 / 1370; N=4 4920 / 3830 / 2670 | `cost_model.py` | analytical formula, inputs \|S\|=100, c0=500, Δ=400, u=30, r=0.1, w=1.25 |
| Table 2 (Haiku), cost column | 13050, 16301, 10059, 34750, 28434, 17216 | `bench/cost_haiku.csv` (`billed_in`) | measured token usage; σ=0 (deterministic) |
| Table 2 (Haiku), hit rate | 0.00, 0.42, 0.37, 0.66 | `bench/cost_haiku.csv` (`hit_rate`) | cache_read / total input tokens |
| Table 2 (Haiku), Rel. cost | 1.00, 1.249, 0.77, 0.82, 0.50 | `bench/cost_haiku.csv` | `billed_in[arch] / billed_in[no_cache]` at same N |
| Abstract / H3 (inversion) | 24.9% | `bench/cost_haiku.csv` | `(billed_in[phase]/billed_in[no_cache] − 1)·100` at N=2 |
| Sonnet comparison | phase 1.25; unified 0.77→0.50 | `bench/cost_sonnet.csv` | `billed_in` ratios (smaller-context run) |
| Cacheable floor | Haiku caches nothing at c0≈1500 (hit 0); Sonnet caches | `bench/floor_haiku.csv`, `bench/cost_sonnet.csv` | `hit_rate` == 0 (Haiku) vs > 0 (Sonnet) |
| Table 3 (H4), correctness | C 0.97 / 0.92; D 1.00 / 1.00 | `results/raw_full.jsonl` + `tasks.json` | mean `score_C/score_D` over all rows (n=108) and the hard subset (`id` prefix `h_`, n=36) |
| Table 3 (H4), pairwise | C 1, D 4; 103/108 ties | `results/raw_full.jsonl` | counts of `pair_winner` |
| H4 non-inferiority CI | 95% CI on D−C = [0.00, 0.08] | `results/raw_full.jsonl` | paired bootstrap over tasks, seed 0, 5000 resamples (same as `ab_experiment.bootstrap_ci`) |
| Task-set composition | 36 items = 24 baseline + 12 hard; 3 repeats | `tasks.json` | count of ids (hard = `h_` prefix) |

**Rounding.** The paper's tables round half-**up** (e.g. 10058.5 → 10059). Python's `round()` is
half-to-even and does not reproduce the page, so `reproduce_paper_numbers.py` uses a `rhu()`
helper. The Rel. cost / inversion % are computed from the raw `billed_in` values (not from the
rounded integer cost cells), which is why the N=2 phase cell reads 1.249 (24.9%), not 16301/13050.

**Worked example vs. measured phase.** The analytical worked example bills the divergent synthesis
turn at *full price* (a simplification for a clean illustration). The measured harness places a
cache breakpoint on every turn, so the divergent turn is billed as a cache *write* (1.25×); this
is why the measured inversion (24.9%) is larger than the worked example's (~9%). Both show the
same qualitative result: a phase-switched loop can cost more than no caching.

## Input numbers (NOT our results — recorded, not reproduced)

These are external facts or provider constants; they are inputs to the paper, not outputs of any
experiment here.

- **Provider pricing / cache multipliers:** read `r ≈ 0.1`, write `w ≈ 1.25` (Anthropic 5-minute
  ephemeral caching); input prices Haiku $1 / Sonnet $3 per 1M (→ the "3×" claim); encoded in
  `bench/cache_prefix_bench.py` (`PRICING`, `CACHE_READ_MULT`, `CACHE_WRITE_MULT`).
- **Cacheable floor ~4096 (Haiku) / ~1024 (Sonnet):** Anthropic-documented; empirically bracketed
  by family 4 (nothing caches on Haiku at c0≈1500; Sonnet caches).
- **Concurrent work:** 41–80% cost reduction, 500+ sessions — Lumer et al. (arXiv:2601.06007) and
  DeepResearch Bench (arXiv:2506.11763). Cited, not measured here.
- **Bibliographic / venue / author metadata:** citation years 2023–2026, "40th … NeurIPS 2026",
  and the author block (name / email / ORCID) live only in the paper, not in this repo.

## Regenerating the raw data (optional, costs API tokens)

The frozen CSVs / JSONL above are the source of truth. To re-measure from scratch:

```bash
# cost / hit-rate tables (deterministic token counts):
python bench/cache_prefix_bench.py --live --model claude-haiku-4-5  --turns 2,4 --repeats 3 --c0-tokens 5000 --delta-tokens 2000 --out bench/cost_haiku.csv
python bench/cache_prefix_bench.py --live --model claude-sonnet-4-6 --turns 2,4 --repeats 3 --c0-tokens 1500 --delta-tokens 2000 --out bench/cost_sonnet.csv
# answer-quality A/B (non-deterministic; live judge, T=0.7):
python ab_experiment.py --repeats 3
```

`c0`/`delta` above are the nominal run settings (the paper reports them as ≈, since measured token
counts differ slightly from the nominal); the committed CSVs are the exact measured record used in
the camera-ready.
