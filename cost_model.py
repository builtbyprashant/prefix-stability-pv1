"""Analytical cost model from the paper (Sec. "Cost Model"). Reproduces the worked example.

All costs are in base-input-token-equivalents (the base input price/token is factored out).
Billed input for a turn = fresh + w * cache_write + r * cache_read, with
    w = 1.25  (5-minute ephemeral cache-write premium),
    r = 0.10  (cached-read rate).
Context grows append-only: turn k (1-indexed) carries context of c0 + (k-1)*delta tokens;
the cacheable prefix at turn k is p_k = |S| + context_k (system + accumulated context).

Architectures:
  no_cache : every turn pays full price on |S| + context_k + u (per-turn instruction).
  unified  : turn 1 writes the whole prefix (|S| + c0); each later turn READS the prior
             prefix, WRITES only the new delta, instruction u fresh. (the fix)
  phase    : identical to unified for the exploration turns; the final synthesis turn uses a
             DIFFERENT system prompt, reuses nothing, and pays full price on the whole
             context |S| + context_N + u. (the pitfall)

Running this file prints the paper's worked example:
  N=2: no_cache=1660  phase=1810  unified=1370
  N=4: no_cache=4920  phase=3830  unified=2670
"""
from __future__ import annotations

W_WRITE = 1.25
R_READ = 0.10


def _billed(fresh: float, write: float, read: float, w: float = W_WRITE, r: float = R_READ) -> float:
    return fresh + w * write + r * read


def worked_example(S: int = 100, c0: int = 500, delta: int = 400, u: int = 30,
                   N: int = 2, w: float = W_WRITE, r: float = R_READ) -> dict:
    """Return {'no_cache', 'phase', 'unified'} billed input (base-token-equivalents) for an
    N-turn loop with the paper's worked-example parameters."""
    ctx = [c0 + k * delta for k in range(N)]                 # turn k (0-indexed) context size

    no_cache = sum(S + ctx[k] + u for k in range(N))

    unified = _billed(u, S + c0, 0, w, r)                    # turn 1: write the prefix
    for k in range(1, N):                                    # later turns: read prior, write delta
        unified += _billed(u, delta, S + ctx[k - 1], w, r)

    phase = _billed(u, S + c0, 0, w, r)                      # turn 1: write the prefix
    for k in range(1, N - 1):                                # middle exploration turns reuse
        phase += _billed(u, delta, S + ctx[k - 1], w, r)
    if N >= 2:                                               # final synthesis turn diverges
        phase += S + ctx[N - 1] + u                          # -> full price, no reuse

    return {"no_cache": round(no_cache), "phase": round(phase), "unified": round(unified)}


if __name__ == "__main__":
    for N in (2, 4):
        m = worked_example(N=N)
        print(f"N={N}: no_cache={m['no_cache']}  phase={m['phase']}  unified={m['unified']}")
