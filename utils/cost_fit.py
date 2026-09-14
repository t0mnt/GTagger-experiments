"""Fixed-plus-marginal cost fit for step timings: ms = fixed + marginal * batch.

A throughput number (jets/s at one batch size) mixes two costs that respond to different
levers: the fixed per-step cost (python, dispatch, kernel launches, host syncs) and the
marginal per-sample cost (arithmetic and memory traffic). rotorch's benchmark doc separates
them with a least-squares line over batch sizes and reads the sparsity win off the marginal
term and the compile win off the fixed term. This is that fit, for the race scripts here.

    python utils/cost_fit.py 32:39.9 128:101.1 512:359.6 2048:1475.4

prints the fixed cost, the marginal cost, and r^2. Batch sizes may be given as jets or as
tokens; the marginal unit follows.
"""

import argparse
import math


def fit(batches, times_ms):
    """Least-squares line through (batch, ms). Returns (fixed_ms, marginal_ms, r2)."""
    if len(batches) != len(times_ms) or len(batches) < 2:
        raise ValueError("need at least two (batch, ms) pairs")
    n = len(batches)
    mx, my = sum(batches) / n, sum(times_ms) / n
    sxx = sum((b - mx) ** 2 for b in batches)
    if sxx == 0:
        raise ValueError("batch sizes must not all be equal")
    sxy = sum((b - mx) * (t - my) for b, t in zip(batches, times_ms))
    marginal = sxy / sxx
    fixed = my - marginal * mx
    ss_res = sum((t - (fixed + marginal * b)) ** 2 for b, t in zip(batches, times_ms))
    ss_tot = sum((t - my) ** 2 for t in times_ms)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return fixed, marginal, r2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", nargs="+", help="batch:ms, e.g. 128:101.1")
    args = ap.parse_args(argv)
    batches, times = zip(*((float(p.split(":")[0]), float(p.split(":")[1])) for p in args.pairs))
    fixed, marginal, r2 = fit(batches, times)
    print(f"fixed {fixed:.2f} ms/step   marginal {marginal * 1e3:.1f} us/sample   r^2 {r2:.4f}")
    if r2 < 0.99 and not math.isclose(r2, 1.0):
        print("  (r^2 < 0.99: the step is not linear in batch size over this range -- "
              "a memory cliff or a compile re-trace is inside it)")


if __name__ == "__main__":
    main()
