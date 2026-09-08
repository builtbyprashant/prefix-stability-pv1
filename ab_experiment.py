"""Answer-quality A/B for the prefix-stability paper (H4: quality parity).

Tests whether relocating the per-turn synthesis instruction OUT of a switched
system prompt (arch C, phase-switched) and INTO the trailing user message
(arch D, unified) preserves final-answer quality. Cost is already settled by the
paper; this isolates quality.

Two complete loop architectures, differing ONLY in where the phase instructions
live (byte-identical instruction text in both arms):

  Arm C (phase-switched):  system prompt CHANGES per phase.
      turn 1 system = ROLE + EXPLORE_GUIDE ; turn 2 system = ROLE + SYNTH_GUIDE.
  Arm D (unified):         system prompt INVARIANT (= ROLE) across the loop.
      per-turn intent rides in the user message (EXPLORE_GUIDE, then SYNTH_GUIDE).

Each arm runs its own 2-turn loop (explore -> synthesize). We pair by
(task, repeat): same task and repeat index -> compare C vs D. A blind, order-
randomized judge (a DIFFERENT, stronger model) grades correctness pointwise and
picks a pairwise winner. Non-inferiority: D is "no worse" than C if the paired
score difference (D - C) has a bootstrap CI lower bound above -delta.

Anthropic has no seed parameter, so "repeats" are independent samples at a fixed
temperature (honest wording for the paper: "R repeats at temperature T").

Run:
  python ab_experiment.py --smoke          # 2 tasks x 1 repeat, prints cost
  python ab_experiment.py --repeats 3      # full run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).parent
GEN_MODEL = "claude-haiku-4-5"      # generator = paper's primary model
JUDGE_MODEL = "claude-sonnet-4-6"   # judge = different, stronger model (no self-preference)
DELTA = 0.05                        # non-inferiority margin (on a 0-1 correctness scale)

# USD per 1M tokens (input, output)
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0)}
_cost = {"gen_in": 0, "gen_out": 0, "judge_in": 0, "judge_out": 0}

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _call(model, system, messages, max_tokens, temperature, retries=5):
    for attempt in range(retries):
        try:
            r = _client.messages.create(model=model, max_tokens=max_tokens,
                                        system=system, messages=messages,
                                        temperature=temperature)
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
        except anthropic.APIError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _gen(system, messages, max_tokens, temperature):
    txt, ti, to = _call(GEN_MODEL, system, messages, max_tokens, temperature)
    _cost["gen_in"] += ti
    _cost["gen_out"] += to
    return txt


def _judge(system, user, max_tokens=200):
    txt, ti, to = _call(JUDGE_MODEL, system, [{"role": "user", "content": user}], max_tokens, 0.0)
    _cost["judge_in"] += ti
    _cost["judge_out"] += to
    return txt


# ── the two architectures ─────────────────────────────────────────────────────

def run_arm(arm, task, cfg, temperature):
    role, explore, synth, nudge = cfg["role_system"], cfg["explore_guide"], cfg["synthesis_guide"], cfg["turn2_nudge"]
    ctx_q = f"Context:\n{task['context']}\n\nQuestion: {task['question']}"
    if arm == "C":  # phase-switched: guides live in the system prompt, which changes
        notes = _gen(role + "\n\n" + explore, [{"role": "user", "content": ctx_q}], 400, temperature)
        msgs = [{"role": "user", "content": ctx_q},
                {"role": "assistant", "content": notes},
                {"role": "user", "content": nudge}]
        ans = _gen(role + "\n\n" + synth, msgs, 300, temperature)
    else:           # D unified: system = ROLE (invariant); guides ride in the user turns
        t1_user = ctx_q + "\n\n" + explore
        notes = _gen(role, [{"role": "user", "content": t1_user}], 400, temperature)
        msgs = [{"role": "user", "content": t1_user},
                {"role": "assistant", "content": notes},
                {"role": "user", "content": nudge + "\n\n" + synth}]
        ans = _gen(role, msgs, 300, temperature)
    return ans.strip(), notes.strip()


# ── judging ───────────────────────────────────────────────────────────────────

_POINT_SYS = ("You are a strict grader for a document-grounded QA task. Given the context, "
              "the question, a reference answer, and a candidate answer, decide if the candidate "
              "is correct. Correct requires ALL of: (a) it conveys the reference answer and is "
              "consistent with the context; (b) for multi-item answers it includes every item the "
              "reference lists with no wrong or missing items, in the requested order if one is "
              "specified; (c) it respects any explicit formatting constraint in the question (e.g. "
              "'only the year', 'one word', 'comma-separated', 'nothing else'). A clear format "
              "violation, a wrong/missing item, or an unrequested reordering is incorrect. If the "
              "reference is 'NOT STATED', the candidate is correct only if it indicates the "
              "information is not in the context. Output ONLY JSON.")

_PAIR_SYS = ("You are a strict, impartial judge for a document-grounded QA task. Given the context, "
             "the question, a reference answer, and two candidate answers (A and B), decide which is "
             "better on correctness, completeness, and following the question's instructions. Output ONLY JSON.")


def _parse(txt, key, allowed):
    s, e = txt.find("{"), txt.rfind("}")
    try:
        v = json.loads(txt[s:e + 1]).get(key)
    except Exception:
        v = None
    if isinstance(v, str):
        v = v.strip().lower()
    return v if v in allowed else None


def judge_point(task, ans):
    user = (f"Context:\n{task['context']}\n\nQuestion: {task['question']}\n\n"
            f"Reference answer: {task['reference']}\n\nCandidate answer: {ans}\n\n"
            'Output JSON: {"correct": true} or {"correct": false}.')
    v = _parse(_judge(_POINT_SYS, user), "correct", {True, False})
    return 1 if v is True else 0  # unparseable -> treat as incorrect (conservative)


def judge_pair(task, ans_c, ans_d, rng):
    swap = rng.random() < 0.5
    a, b = (ans_d, ans_c) if swap else (ans_c, ans_d)
    user = (f"Context:\n{task['context']}\n\nQuestion: {task['question']}\n\n"
            f"Reference answer: {task['reference']}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\n"
            'Which is better? Output JSON: {"better": "A"} or {"better": "B"} or {"better": "tie"}.')
    pick = _parse(_judge(_PAIR_SYS, user), "better", {"a", "b", "tie"})
    if pick is None or pick == "tie":
        return "tie"
    winner_is_a = pick == "a"
    if swap:    # A=D, B=C
        return "D" if winner_is_a else "C"
    return "C" if winner_is_a else "D"  # A=C, B=D


# ── stats ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(per_task_c, per_task_d, n=5000, seed=0):
    """Paired bootstrap over TASKS of mean(D) - mean(C)."""
    rng = random.Random(seed)
    k = len(per_task_c)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        mc = sum(per_task_c[i] for i in idx) / k
        md = sum(per_task_d[i] for i in idx) / k
        diffs.append(md - mc)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--smoke", action="store_true", help="2 tasks x 1 repeat")
    ap.add_argument("--tasks-limit", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    tasks = cfg["tasks"]
    if args.smoke:
        tasks, args.repeats = tasks[:2], 1
    elif args.tasks_limit:
        tasks = tasks[:args.tasks_limit]

    rng = random.Random(1234)
    rows = []
    # per-task accumulators (mean correctness across repeats), for the paired bootstrap
    ptask_c, ptask_d = [], []
    wins = {"C": 0, "D": 0, "tie": 0}
    t0 = time.time()

    for ti, task in enumerate(tasks):
        c_scores, d_scores = [], []
        for r in range(args.repeats):
            ans_c, notes_c = run_arm("C", task, cfg, args.temperature)
            ans_d, notes_d = run_arm("D", task, cfg, args.temperature)
            sc = judge_point(task, ans_c)
            sd = judge_point(task, ans_d)
            pair = judge_pair(task, ans_c, ans_d, rng)
            wins[pair] += 1
            c_scores.append(sc)
            d_scores.append(sd)
            rows.append({"task": task["id"], "repeat": r, "score_C": sc, "score_D": sd,
                         "pair_winner": pair, "ans_C": ans_c, "ans_D": ans_d})
            print(f"  [{ti+1}/{len(tasks)}] {task['id']} r{r}: C={sc} D={sd} pair={pair}", flush=True)
        ptask_c.append(sum(c_scores) / len(c_scores))
        ptask_d.append(sum(d_scores) / len(d_scores))

    mean_c = sum(ptask_c) / len(ptask_c)
    mean_d = sum(ptask_d) / len(ptask_d)
    diff = mean_d - mean_c
    lo, hi = bootstrap_ci(ptask_c, ptask_d)
    non_inferior = lo > -DELTA

    gi, go = _cost["gen_in"], _cost["gen_out"]
    ji, jo = _cost["judge_in"], _cost["judge_out"]
    usd = (gi * PRICE[GEN_MODEL][0] + go * PRICE[GEN_MODEL][1]
           + ji * PRICE[JUDGE_MODEL][0] + jo * PRICE[JUDGE_MODEL][1]) / 1e6

    n_pairs = wins["C"] + wins["D"]
    summary = {
        "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
        "n_tasks": len(tasks), "repeats": args.repeats, "temperature": args.temperature,
        "mean_correct_C": round(mean_c, 4), "mean_correct_D": round(mean_d, 4),
        "diff_D_minus_C": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "delta": DELTA, "non_inferior": non_inferior,
        "pairwise": wins,
        "pairwise_D_pref_excl_ties": round(wins["D"] / n_pairs, 4) if n_pairs else None,
        "cost_usd": round(usd, 4), "elapsed_s": round(time.time() - t0, 1),
    }

    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    (outdir / f"raw_{tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (outdir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nInterpretation: D (unified) mean correctness {mean_d:.3f} vs C (phase-switched) "
          f"{mean_c:.3f}; diff {diff:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}].")
    print("Non-inferiority (D no worse than C within delta="
          f"{DELTA}): {'PASS' if non_inferior else 'NOT ESTABLISHED'}.")


if __name__ == "__main__":
    main()
