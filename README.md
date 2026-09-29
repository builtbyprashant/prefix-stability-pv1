# Prefix Stability for Prompt Caching in LLM Agents
**When It Helps and When It Backfires**

A study of a small, common design choice in AI agents that can **silently multiply your bill**, sometimes making a "cost-saving" feature cost *more* than switching it off. This repository is both the paper's companion and its **single source of truth**: one command regenerates every result number in the paper (see [Reproduce](#part-6-reproduce-every-number-yourself)).

*Accepted at the GlobalSouthAI workshop, NeurIPS 2026. This repo is a private, de-identified reproduction snapshot frozen at the accepted-paper state.*

---

## What this is, in 30 seconds

Modern AI "agents" work by calling a language model over and over in a loop, each time re-sending everything said so far. That repetition is expensive. Providers offer **prompt caching** to make it cheap, but caching only works if each request begins with the *exact same text* as a previous one. A very natural way to build an agent, using a different instruction ("system prompt") for different stages of its work, quietly breaks that condition. The result: no error, a correct answer, and a bill that is **much larger than expected**, sometimes larger than if you had never turned caching on.

This paper explains **exactly when caching helps, when it silently fails, and when it backfires**, backs it with a cost model and live measurements on production models, gives a **one-line rule** to avoid the trap, and shows a **free way to detect** it from a field the API already returns in its ordinary usage payload.

---

## Contents

- [Part 1: The idea, in plain language](#part-1-the-idea-in-plain-language)
- [Part 2: The four ways to build the loop](#part-2-the-four-ways-to-build-the-loop)
- [Part 3: Why it costs what it costs](#part-3-why-it-costs-what-it-costs)
- [Part 4: What we measured](#part-4-what-we-measured)
- [Part 5: The one rule, and how to check your own agent](#part-5-the-one-rule-and-how-to-check-your-own-agent)
- [What's new here, and what isn't](#whats-new-here-and-what-isnt)
- [Part 6: Reproduce every number yourself](#part-6-reproduce-every-number-yourself)
- [Methods (for the technical reader)](#methods-for-the-technical-reader)
- [Limitations and honest scope](#limitations-and-honest-scope)
- [Glossary](#glossary)
- [Repository layout](#repository-layout)
- [Citation](#citation)

---

## Part 1: The idea, in plain language

### What is an "agent loop"?

A language model (LLM) reads some text and writes a continuation. An **agent** uses it in a loop: think, maybe use a tool, read the result, think again, and finally answer. Because the model has no memory between calls, the program must **re-send the entire conversation so far on every turn**. Turn 1 sends a little; turn 5 re-sends everything from turns 1–4 plus the new bit. The text you send is the **input**, and you pay per input token (a token ≈ ¾ of a word).

So the input grows every turn, and you pay for the whole thing every turn. Over an *N*-turn loop that means paying for early context again and again: turn 1 re-sends a little, the last turn re-sends nearly everything, and adding up N steadily growing turns lands the total at roughly **N²** (the square of the number of turns).

### What is "prompt caching"?

To avoid re-charging full price for text the model has already processed, providers offer **prompt caching**. The mental model:

> The model keeps a "warm" copy of a chunk of text it recently processed. If your next request **begins with the byte-for-byte identical chunk**, it reuses the warm copy and charges you about **one tenth** of the normal price for it, instead of full price.

There is a catch that turns out to matter a lot: the match must start at the very beginning and be **exact**. Change one character near the front and the cache can't help you past that point, because everything after a change is different too. There is also a small **write fee** (about 1.25× normal) the first time a chunk is stored, and a **minimum size**: chunks below a provider- and model-specific floor are ignored silently.

### The trap: a silent, expensive mistake

A request to an agent is, in order: *model id · tools · **system prompt** · conversation so far*. The **system prompt** is the standing instruction ("You are a research assistant…") and it sits near the **front**.

Many agents change the system prompt between phases, e.g. one instruction while *exploring* and a different one while *writing the final answer*. It feels harmless. But because the system prompt is near the front, **swapping it changes the text early**, so the new request no longer matches the cached copy of everything that came after. On the turn where the context is **largest** (the final synthesis), you get **no cache discount at all**.

Here is the part that surprises people. Along the way you already paid the small **write fee** to store those chunks, betting you'd reuse them cheaply later. If the final, biggest turn can't reuse them, that bet is lost. For short loops the wasted write fees can add up to **more than you would have paid with no caching at all**. The feature meant to save money loses it, with no error and a perfectly correct answer to hide it.

**Analogy.** You're photocopying a growing stack of pages each round. The copier offers a deal: it remembers pages it already scanned and charges 10% to reprint them, *as long as page 1 is unchanged*, plus a small fee to memorize each page. If you swap page 1 on the final, biggest round, it must rescan everything at full price, and you're still stuck with all those "memorize" fees. You'd have been better off never using the deal.

---

## Part 2: The four ways to build the loop

Every real design is one of four patterns. Only the last one is safe.

| | Architecture | What it does | What it costs |
|---|---|---|---|
| **A** | **No caching** | Full prompt re-sent every turn. | Baseline; you pay full price every turn. |
| **B** | **System cached** | Caches only the (short) system prompt. It's below the minimum size, so the provider **silently ignores it**. | Identical to A, you *think* you're saving but you're not. |
| **C** | **Phase-switched** | Caches the conversation, but the **system prompt changes** on the final turn. | Same-phase turns get a discount; the biggest turn gets none, and the wasted write fees can make it **cost *more* than A**. |
| **D** | **Unified, append-only** | **One** system prompt for the whole loop; per-turn instructions ride at the **end** (each turn only *adds* text, never editing what came before), so changing them never disturbs the cached front. | The only pattern that keeps the discount across the whole loop. |

The fix that turns a broken **C** into a healthy **D** is tiny: stop switching the system prompt, keep it constant, and move the "now do X" instruction from the system prompt to the **last user message**. Same words, different place.

---

## Part 3: Why it costs what it costs

**The intuition.** Without caching, each turn re-pays full price for all the old context, so total cost grows like **N²**. With a stable, append-only prefix (pattern D), each turn only *reads* the old context cheaply and *writes* the small new piece, so the **full-price work grows only like N**. The catch: those cheap re-reads still pile up, so the *total* bill keeps a discounted N² term and settles at roughly **10%** of the uncached bill for long loops, not zero. Big win, but not free.

**The formal results** (full derivations in the paper):

- **Full-rate collapse.** Under an append-only context and an unchanging system prompt, the full-price input work in an *N*-turn loop is **Θ(N)** (grows about linearly with N) instead of **Θ(N²)** (grows with the square of N).
- **Caching can invert (the backfire).** Because the write fee is greater than 1× (about 1.25×), there exist realistic setups (short loop, phase-switched system) where **turning caching on costs more than leaving it off**.

**A worked example you can check by hand.** With a 100-token system prompt, 500 tokens of starting context, 400 new tokens per turn, a 30-token per-turn instruction, read rate 0.1×, write rate 1.25× (all costs in "base-input-token" units):

| Turns | A: no caching | C: phase-switched | D: unified |
|------:|:-------------:|:-----------------:|:----------:|
| N = 2 | 1660 | **1810** (worse!) | **1370** (best) |
| N = 4 | 4920 | 3830 | **2670** |

At **N = 2**, phase-switching (C) costs **more** than no caching (A), the inversion in the flesh. Unified (D) is cheapest at both lengths, and its advantage grows with the loop. (`cost_model.py` reproduces these exactly.)

This toy model bills the divergent final turn at full price, a ~9% inversion here. The live measurements in [Part 4](#part-4-what-we-measured) place a cache marker on every turn, so that turn is billed as a 1.25× cache *write*, which is why the **measured** inversion is larger (24.9%). Same phenomenon; the measurement simply captures the write premium the illustration leaves out.

---

## Part 4: What we measured

We ran all four patterns against production models and read the token counts straight from each API response. Token counts are deterministic, so these numbers are exact and repeatable (not noisy averages).

**Main result (Claude Haiku, growing context).** "Rel. cost" is the bill relative to no-caching (A) for the same loop length; lower is better. "Hit rate" is the share of input tokens served cheaply from cache.

| N | Architecture | Hit rate | Cost (base-tok) | Rel. cost |
|--:|---|:--:|--:|:--:|
| 2 | A no-cache | 0.00 | 13050 | 1.00 |
| 2 | B system-cached | 0.00 | 13050 | 1.00 |
| 2 | C phase-switched | 0.00 | 16301 | **1.249** |
| 2 | D unified | 0.42 | 10059 | **0.77** |
| 4 | A no-cache | 0.00 | 34750 | 1.00 |
| 4 | B system-cached | 0.00 | 34750 | 1.00 |
| 4 | C phase-switched | 0.37 | 28434 | 0.82 |
| 4 | D unified | 0.66 | 17216 | **0.50** |

Reading this table:

- **The backfire is real.** At N = 2, phase-switching (C) costs **24.9% more** than no caching (A). B is identical to A, confirming that caching only the sub-floor system prompt does nothing.
- **The win grows.** Unified (D) falls from 0.77 to 0.50 of the baseline as the loop lengthens, exactly the Θ(N²)→Θ(N) effect.
- **The tell.** C's hit rate is **0.00 at N = 2**, it never reads across the phase switch. A zero cache-read where you expected reuse is the fingerprint of the bug.

**It's not one model or one price.** Repeating on a stronger, 3× more expensive model (Claude Sonnet) at a smaller context, the *relative* costs match Haiku to two digits (phase inverts to **1.25**; unified falls **0.77 → 0.50**). The effect is driven by the shared rate structure, not the absolute price. The **minimum cache size is model-specific**, though: at ~1500 tokens of context, the unified pattern caches **nothing** on Haiku (floor ~4096) yet caches readily on Sonnet (floor ~1024), same code, different result.

**Does the fix hurt answer quality?** The fix moves the synthesis instruction from the system prompt into the last user message, so we checked that answers don't get worse. On 36 grounded question-answering items, each run 3 times (108 answers per pattern), graded by a *separate, stronger* model that never saw which pattern produced which answer (a blind judge), the unified pattern (D) **matched or beat** the phase-switched one (C). Correctness was C 0.97 / D 1.00 across all 108, and C 0.92 / D 1.00 on the 36 hardest items; head-to-head, D won 4, C won 1, and **103 of 108** tied. A confidence interval is the range the true difference plausibly lies in; on D − C it was **[0.00, 0.08]**, entirely at or above zero (D never came out worse), so there is no evidence the fix lowers quality. The one systematic difference favored D (the phase-switched arm over-listed a year on an aggregation question). The cost win is not paid for in quality.

---

## Part 5: The one rule, and how to check your own agent

**The rule.**

> Use a **single, unchanging system prompt** for the whole loop. Put per-turn intent in the **trailing user message**, never by switching the system prompt.

**Three checks that predict, before you deploy, whether reuse survives the whole loop:**

1. Is the cached chunk **above the model's minimum size**? (Measure it; the floor differs by model.)
2. Is the system prompt **byte-identical every turn**?
3. Does each turn's prefix **extend the previous one exactly**, with the cache markers left in place (the flags on a request that tell the provider which leading chunk to cache; Anthropic calls them cache *breakpoints*)?

Only pattern **D** passes all three.

**The free detector.** Every response reports `cache_read_input_tokens`. On any turn where you expected to reuse earlier context, a **zero** there means a check failed, your cache isn't firing, and you're overpaying. No special tooling, profiler, or vendor support needed; the signal ships in the ordinary API response. That low barrier is the point: any team, however small its budget, can audit this.

---

## What's new here, and what isn't

The one-line rule ("keep the system prompt fixed, put per-turn intent last") is **not new**: it is an engineering heuristic that both Anthropic and OpenAI already recommend. What this paper contributes is the **formal, measured account** behind it: a cost model with the Θ(N²)→Θ(N) collapse, a proof that misusing caching can *invert* into a net cost increase, a four-architecture failure taxonomy with a pre-deployment decision procedure, and a telemetry-based measurement protocol. It is **complementary** to concurrent work that quantifies caching's aggregate benefit at scale (Lumer et al., *Don't Break the Cache*, arXiv:2601.06007; DeepResearch Bench, arXiv:2506.11763: 500+ sessions, 41–80% cost reduction): that line of work measures how much caching saves across many sessions, while we isolate *why* one common architectural choice defeats the critical cross-phase reuse and show the failure can cost more than no caching at all.

## Part 6: Reproduce every number yourself

This repo is the **single source of truth** for the paper. One command regenerates every quoted number from the frozen data here and asserts it matches the paper:

```bash
python reproduce_paper_numbers.py      # -> "46/46 reproducible numbers match", exit 0
```

Pure Python standard library, **no API key, no network, no installs**. Verified from a clean clone.

What it checks, and from where:

| Family | Numbers | Source |
|---|---|---|
| Analytical worked example | 1660 / 1810 / 1370, 4920 / 3830 / 2670 | `cost_model.py` (a formula) |
| Cost & hit-rate, Haiku | the main results table, the 24.9% backfire | `bench/cost_haiku.csv` |
| Cost, Sonnet | 1.25, 0.77 → 0.50 | `bench/cost_sonnet.csv` |
| Cache floor | Haiku caches nothing / Sonnet caches at small context | `bench/floor_haiku.csv` + `bench/cost_sonnet.csv` |
| Answer quality | correctness, ties, the [0.00, 0.08] interval | `results/raw_full.jsonl` + `tasks.json` |

A number-by-number provenance map, including the handful of values that are **inputs, not our results** (provider prices and cited work, which are recorded with their sources rather than "reproduced"), is in **[`PAPER_NUMBERS.md`](PAPER_NUMBERS.md)**.

> **Rounding note.** The paper's tables round half **up** (e.g. 10058.5 → 10059); Python's built-in `round()` rounds half to even and would not match the page, so the script uses a matching helper. Relative costs are computed from the raw token bills, not from the rounded cells.

To re-measure from scratch (costs a few API cents; the committed data is authoritative), see the commands at the end of `PAPER_NUMBERS.md`.

---

## Methods (for the technical reader)

- **Cost harness** (`bench/cache_prefix_bench.py`). Drives a fixed-length loop under all four architectures and records `input_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens` from the provider usage payload. Loop length is fixed on purpose: this is an **input-side** effect, so holding the model's decisions constant isolates the architecture variable. Billed input = `fresh + 1.25·writes + 0.1·reads` (base input price factored out); with Anthropic 5-minute ephemeral caching, one breakpoint at the end of the stable prefix, calls within the TTL. Token counts are deterministic (σ = 0). Defaults to a no-spend analytical dry-run; `--live` measures against the API.
- **Quality A/B** (`ab_experiment.py`). Two 2-turn loops (explore → synthesize) differing in **exactly one thing**, whether per-turn intent lives in a switched system prompt (C) or the trailing user message (D); the instruction text is byte-identical across arms. Generator: `claude-haiku-4-5` at temperature 0.7. Judge: `claude-sonnet-4-6` (a different, stronger model, to avoid self-preference), grading pointwise correctness and a blind, order-randomized pairwise winner. Metric is **non-inferiority**, not superiority: paired difference `mean(D) − mean(C)` with a 95% paired bootstrap CI over tasks (seed 0, 5000 resamples, so it recomputes exactly), non-inferiority margin 0.05.
- **Task set** (`tasks.json`). 36 document-grounded QA items over three public encyclopedia articles: 24 baseline factual items plus 12 harder ones (strict-format, aggregation/completeness, light multi-hop; ids prefixed `h_`). Several items are unanswerable from the context to probe instruction-following, not just recall.

---

## Limitations and honest scope

- Measurements use two Anthropic models and a synthetic, fixed-length loop. The qualitative claims should transfer to other providers (automatic vs. explicit caching, radix/block reuse), but the **thresholds must be re-measured**, they are provider- and model-specific.
- The quality result is "no evidence of degradation" on short synthetic loops with a single-model judge; parity on long-horizon, multi-tool tasks with heavier synthesis prompts is not yet stressed.
- Branching agents (tree search, multi-agent) are out of scope; the taxonomy should extend but the cost expressions would need per-branch accounting.
- Findings are within a provider's cache lifetime (TTL). This is a cost/efficiency study, not a claim about answer quality in general.

---

## Glossary

- **Token**: the unit models read and bill by; ≈ ¾ of a word.
- **Prompt / context**: the text you send; for an agent it grows every turn.
- **System prompt**: the standing instruction near the front of the request.
- **Prefix**: the leading stretch of a request; caching matches on the prefix.
- **Prompt caching**: reusing an already-processed prefix at a steep discount (~0.1×), with a small write fee (~1.25×) and a minimum cacheable size.
- **Hit rate**: the share of input tokens served from cache; 0 means the cache never fired.
- **Θ(N) / Θ(N²)**: "grows about linearly / about with the square of" the number of turns N.
- **Non-inferiority**: a test that something is *no worse* (within a margin), not necessarily better.
- **Confidence interval**: the range a true value plausibly falls in; if the whole range for "D minus C" sits at or above zero, D is not worse.
- **Append-only**: a context that only ever grows at the end; earlier text is never edited, so the cached front stays byte-identical.
- **Cache marker / breakpoint**: a flag on a request telling the provider which leading chunk to cache; Anthropic calls it a cache breakpoint.
- **TTL (time to live)**: how long a cached prefix stays reusable before it expires; Anthropic ephemeral caching uses ~5 minutes.

---

## Repository layout

```
reproduce_paper_numbers.py   # regenerates + asserts every paper number (start here)
PAPER_NUMBERS.md             # each paper number -> its source and formula
cost_model.py                # analytical cost model (the worked example)
bench/
  cache_prefix_bench.py      # cost / hit-rate measurement harness (4 architectures)
  cost_haiku.csv             # main results table (Haiku)       : measured, deterministic
  cost_sonnet.csv            # cross-model check (Sonnet)        : measured, deterministic
  floor_haiku.csv            # cache-floor demo (Haiku, small context)
ab_experiment.py             # answer-quality A/B harness (H4)
tasks.json                   # 36 QA items (24 baseline + 12 hard, id prefix `h_`)
results/
  raw_full.jsonl             # every answer + judgement          : frozen
  summary_full.json          # means, difference, CI, pairwise   : frozen
  raw_smoke.jsonl / summary_smoke.json
```

The API key, when re-measuring, is read from the `ANTHROPIC_API_KEY` environment variable and never hardcoded. Kept private and separate from the public `genai-doc-assistant` repository; the corpus is three public encyclopedia articles (AI, AGI, ML).

---

## Citation

Paper: *Prefix Stability for Prompt Caching in LLM Agents: When It Helps and When It Backfires.* GlobalSouthAI workshop, NeurIPS 2026. The full author block and canonical citation are in the published camera-ready; this repository is kept de-identified by design.
