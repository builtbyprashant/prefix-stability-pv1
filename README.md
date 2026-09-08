# Answer-quality A/B (H4: quality parity)

Supplementary experiment for *Prefix Stability for Prompt Caching in LLM Agents*.
The paper proves and measures the **cost** difference between a phase-switched
loop (arch C) and a unified, append-only loop (arch D). This experiment tests the
remaining question: does moving the per-turn synthesis instruction out of a
switched system prompt (C) and into the trailing user message (D) **preserve
answer quality**? If not, the cost win is hollow.

## Design

Two complete 2-turn loops (explore -> synthesize) that differ in **exactly one
thing**: where the phase instructions live. The instruction *text* is byte-identical
across arms.

| | System prompt | Per-turn intent |
|---|---|---|
| **Arm C** (phase-switched) | changes per phase (`ROLE+EXPLORE`, then `ROLE+SYNTH`) | in the system prompt |
| **Arm D** (unified) | invariant (`ROLE`) across the loop | in the user message |

- **Task set:** 36 document-grounded QA items (`tasks.json`) drawn from the three
  public Wikipedia sample documents shipped with the app (AI, AGI, ML): 24 baseline
  factual items plus 12 harder items (strict-format single facts, aggregation /
  completeness lists, and light multi-hop) that give the grader headroom. Short
  reference answers; several items are unanswerable from the context (reference
  `NOT STATED`) to probe instruction-compliance, not just recall.
- **Generator:** `claude-haiku-4-5` (the paper's primary model), temperature 0.7.
- **Repeats:** 3 independent samples per (task, arm). The Anthropic API has no seed
  parameter, so repeats are resamples at fixed temperature (reported honestly as
  "3 repeats at T=0.7"), which is also what the cost tables use.
- **Judge:** `claude-sonnet-4-6` — a *different, stronger* model, to avoid
  self-preference. Two judgements per (task, repeat): (1) pointwise correctness of
  each answer vs the reference (0/1); (2) a blind, order-randomized pairwise winner
  (C / D / tie).

## Metric — non-inferiority, not superiority

We are showing D is **no worse** than C. Primary metric: paired difference in mean
correctness, `mean(D) - mean(C)`, with a 95% paired bootstrap CI resampled over
tasks. Pre-specified non-inferiority margin `delta = 0.05`: parity holds if the CI
lower bound is above `-delta`. Pairwise win/tie/loss is reported as a secondary,
judge-model-independent check.

## Run

```
python ab_experiment.py --smoke        # 2 tasks x 1 repeat, prints cost
python ab_experiment.py --repeats 3    # full run, 36 tasks (~$0.7, ~22 min)
```

Outputs land in `results/`: `raw_full.jsonl` (every answer + judgement) and
`summary_full.json` (means, diff, CI, pairwise, cost).

## Reproducibility / anonymity note

The paper is under double-blind review, so it must **not** link to the public
`genai-doc-assistant` repository or any identifying account. This folder is kept
outside that repo; if the result is added to the paper, describe the corpus as
"three public encyclopedia articles" without a repo link until the review is over.
