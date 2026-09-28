"""Single source of truth for the camera-ready paper
"Prefix Stability for Prompt Caching in LLM Agents: When It Helps and When It Backfires".

Regenerates EVERY number quoted in the paper from this repo's frozen data and asserts each
one matches the value printed in the camera-ready. Exit code 0 iff every reproducible number
matches; non-zero otherwise.

    python reproduce_paper_numbers.py

Number families and their single source:
  1. Analytical worked example    -> cost_model.py                          (a formula, no data)
  2. Cost / hit-rate Table 2      -> bench/cost_haiku.csv                    (measured, sigma=0)
  3. Sonnet cost comparison       -> bench/cost_sonnet.csv                   (measured, sigma=0)
  4. Cacheable-floor claims       -> bench/floor_haiku.csv + bench/cost_sonnet.csv
  5. Answer-quality Table 3 / H4  -> results/summary_full.json, results/raw_full.jsonl, tasks.json

Numbers that are NOT our results (external citations and provider constants) are inputs, not
reproducible outputs; they are listed at the end and mapped in PAPER_NUMBERS.md.

Rounding note: the paper's tables round half-UP; Python's round() is half-to-even and does not
reproduce cells like 10058.5 -> 10059. We use rhu() (round half up) to match the printed page.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cost_model

HERE = Path(__file__).parent
_results = []  # (ok, label, got_str, paper_str)


def rhu(x: float, n: int = 0) -> float:
    """Round half up to n decimals (matches the paper's table rounding)."""
    f = 10 ** n
    return math.floor(x * f + 0.5) / f


def check(label: str, got: float, paper, prec: int = 2, unit: str = "") -> bool:
    got_s = f"%.{prec}f" % rhu(got, prec)
    paper_s = f"%.{prec}f" % float(paper)
    ok = got_s == paper_s
    _results.append((ok, label, got_s + unit, paper_s + unit))
    return ok


def check_int(label: str, got: float, paper, unit: str = "") -> bool:
    got_i = int(rhu(got, 0))
    ok = got_i == int(paper)
    _results.append((ok, label, f"{got_i}{unit}", f"{int(paper)}{unit}"))
    return ok


def note(ok: bool, label: str, got: str, paper: str):
    _results.append((ok, label, got, paper))


def load_cost(fn: str) -> dict:
    rows = {}
    with open(HERE / "bench" / fn, newline="") as f:
        for r in csv.DictReader(f):
            rows[(int(r["N"]), r["arch"])] = r
    return rows


def rel(rows, N, arch):
    return float(rows[(N, arch)]["billed_in"]) / float(rows[(N, "no_cache")]["billed_in"])


# ============ 1. Worked example (analytical) ============
for N, exp in ((2, (1660, 1810, 1370)), (4, (4920, 3830, 2670))):
    m = cost_model.worked_example(N=N)
    check_int(f"worked N={N} no_cache", m["no_cache"], exp[0])
    check_int(f"worked N={N} phase", m["phase"], exp[1])
    check_int(f"worked N={N} unified", m["unified"], exp[2])

# ============ 2. Haiku Table 2 ============
H = load_cost("cost_haiku.csv")
exp_cost = {(2, "no_cache"): 13050, (2, "sys_cached"): 13050, (2, "phase"): 16301, (2, "unified"): 10059,
            (4, "no_cache"): 34750, (4, "sys_cached"): 34750, (4, "phase"): 28434, (4, "unified"): 17216}
for key, v in exp_cost.items():
    check_int(f"haiku cost {key}", float(H[key]["billed_in"]), v)
exp_hit = {(2, "no_cache"): 0.00, (2, "sys_cached"): 0.00, (2, "phase"): 0.00, (2, "unified"): 0.42,
           (4, "no_cache"): 0.00, (4, "sys_cached"): 0.00, (4, "phase"): 0.37, (4, "unified"): 0.66}
for key, v in exp_hit.items():
    check(f"haiku hit {key}", float(H[key]["hit_rate"]), v, 2)
check("haiku rel (2,phase)", rel(H, 2, "phase"), 1.249, 3)
check("haiku rel (2,unified)", rel(H, 2, "unified"), 0.77, 2)
check("haiku rel (4,phase)", rel(H, 4, "phase"), 0.82, 2)
check("haiku rel (4,unified)", rel(H, 4, "unified"), 0.50, 2)
check("haiku inversion pct (N=2)", (rel(H, 2, "phase") - 1) * 100, 24.9, 1, "%")

# ============ 3. Sonnet ============
S = load_cost("cost_sonnet.csv")
check("sonnet rel (2,phase)", rel(S, 2, "phase"), 1.25, 2)
check("sonnet rel (2,unified)", rel(S, 2, "unified"), 0.77, 2)
check("sonnet rel (4,unified)", rel(S, 4, "unified"), 0.50, 2)

# ============ 4. Cacheable floor ============
F = load_cost("floor_haiku.csv")
haiku_floor_ok = all(float(F[(N, "unified")]["hit_rate"]) == 0.0 for N in (2, 4))
note(haiku_floor_ok, "floor: Haiku unified caches nothing at small ctx (hit==0)", "0.00", "0.00")
sonnet_hit = float(S[(2, "unified")]["hit_rate"])
note(sonnet_hit > 0, "floor: Sonnet unified caches at small ctx (hit>0)", f"{sonnet_hit:.2f}", ">0")

# ============ 5. Answer quality (H4) / Table 3 ============
raw = [json.loads(l) for l in (HERE / "results" / "raw_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
task_ids = [t["id"] for t in tasks]
hard_ids = {t["id"] for t in tasks if t["id"].startswith("h_")}

check_int("H4 tasks", len(tasks), 36)
check_int("H4 hard items", len(hard_ids), 12)
check_int("H4 baseline items", len(tasks) - len(hard_ids), 24)
check_int("H4 rows (item x repeat)", len(raw), 108)

check("Table3 C correct (all, n=108)", sum(r["score_C"] for r in raw) / len(raw), 0.97, 2)
check("Table3 D correct (all, n=108)", sum(r["score_D"] for r in raw) / len(raw), 1.00, 2)
hard = [r for r in raw if r["task"] in hard_ids]
check_int("H4 hard rows", len(hard), 36)
check("Table3 C correct (hard, n=36)", sum(r["score_C"] for r in hard) / len(hard), 0.92, 2)
check("Table3 D correct (hard, n=36)", sum(r["score_D"] for r in hard) / len(hard), 1.00, 2)

pw = Counter(r["pair_winner"] for r in raw)
check_int("Table3 pairwise C wins", pw["C"], 1)
check_int("Table3 pairwise D wins", pw["D"], 4)
check_int("H4 ties (of 108)", pw["tie"], 103)

# 95% paired bootstrap CI over tasks, seed=0, n=5000 (identical to ab_experiment.bootstrap_ci)
bytask = defaultdict(lambda: [[], []])
for r in raw:
    bytask[r["task"]][0].append(r["score_C"])
    bytask[r["task"]][1].append(r["score_D"])
ptc = [sum(bytask[t][0]) / len(bytask[t][0]) for t in task_ids]
ptd = [sum(bytask[t][1]) / len(bytask[t][1]) for t in task_ids]
rng = random.Random(0)
k = len(ptc)
diffs = []
for _ in range(5000):
    idx = [rng.randrange(k) for _ in range(k)]
    diffs.append(sum(ptd[i] for i in idx) / k - sum(ptc[i] for i in idx) / k)
diffs.sort()
lo, hi = diffs[int(0.025 * 5000)], diffs[int(0.975 * 5000)]
check("H4 CI lower (D-C)", lo, 0.00, 2)
check("H4 CI upper (D-C)", hi, 0.08, 2)

# ============ report ============
npass = sum(1 for r in _results if r[0])
print("Reproducing every camera-ready number from this repo:\n")
for ok, label, got, paper in _results:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:38s} repo {got:>9s}   paper {paper:>9s}")
print(f"\n{npass}/{len(_results)} reproducible numbers match the camera-ready.")

print("\nExternal citations & provider constants (INPUTS, not our outputs; see PAPER_NUMBERS.md):")
for s in (
    "cache read r=0.1, write w=1.25  (Anthropic 5-min ephemeral pricing)",
    "cacheable floor ~4096 Haiku / ~1024 Sonnet  (Anthropic docs; empirically bracketed by family 4)",
    "Sonnet vs Haiku input price 3x  ($3 vs $1 per 1M input tokens)",
    "41-80% cost reduction, 500+ sessions  (Lumer et al., arXiv:2601.06007; DeepResearch Bench, arXiv:2506.11763)",
    "citation years 2023-2026; venue '40th ... NeurIPS 2026'; author metadata (name/email/ORCID) in the paper's author block",
):
    print("   -", s)

sys.exit(0 if npass == len(_results) else 1)
