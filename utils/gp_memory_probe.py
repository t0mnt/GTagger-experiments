"""Retention AND peak per `gp_impl`, eager and compiled (docs/cgenn-compile.md).

    python utils/gp_memory_probe.py        # ~3 min on CPU; WARMUP=n to change warm-ups


The load-bearing claim for the campaign posture -- "under torch.compile AOT's partitioner
equalizes retention across impls" -- is asserted in three config comments, two docstrings and
the CORRECTION block, and gated NOWHERE. Re-measure it.

Warm-up matters: the original 5.6x figure was wrong precisely because it was read on the
first call, i.e. during compilation.
"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # NOT REPO/tests -- that shadows the `experiments` package
os.chdir(REPO)

import torch

torch.set_num_threads(1)  # CPU inductor + OpenMP aborts CGENN otherwise (audit ledger)

from tests.experiments.test_cgenn_compile import FIX, _build, _rebuild, _saved_bytes


def peak_bytes(fn):
    """Max live tensor bytes during `fn`, via the profiler's allocation records.

    Retention is not peak: the partitioner can equalise what CROSSES the fwd/bwd boundary
    while the impls still differ in transients, and it is PEAK that sets the batch size.
    Under compile a TorchDispatchMode sees nothing (the graph is one call), so go through
    the profiler, which records inductor's own allocations too.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    live = hi = 0
    for e in prof.events():
        delta = getattr(e, "cpu_memory_usage", 0) or 0
        live += delta
        hi = max(hi, live)
    return hi

ref = torch.load(FIX / "fp64.pt", weights_only=False)
WARMUP = int(os.environ.get("WARMUP", 3))

MB = 2**20
print(f"{'impl':>8} {'retained eager':>15} {'retained comp':>14} "
      f"{'peak eager':>12} {'peak comp':>11}", flush=True)
rows = {}
for impl in ("einsum", "matmul", "sparse"):
    ov = [f"model.net.gp_impl={impl}"]

    exp = _build(float64=True, extra_overrides=ov)
    exp.model.load_state_dict(ref["sd"], strict=True)
    eager_ret = _saved_bytes(exp, _rebuild(ref["batch"]))
    eager_peak = peak_bytes(lambda: exp._get_ypred_and_label(_rebuild(ref["batch"]))[0].sum().backward())

    exp = _build(float64=True, extra_overrides=ov)
    exp.model.load_state_dict(ref["sd"], strict=True)
    exp.model.net = torch.compile(exp.model.net, dynamic=True)
    exp.model.train()
    t0 = time.perf_counter()
    for _ in range(WARMUP):  # first call is compilation; anything read then is meaningless
        exp._get_ypred_and_label(_rebuild(ref["batch"]))[0].sum().backward()
    warm = time.perf_counter() - t0
    comp_ret = _saved_bytes(exp, _rebuild(ref["batch"]))
    comp_peak = peak_bytes(lambda: exp._get_ypred_and_label(_rebuild(ref["batch"]))[0].sum().backward())

    rows[impl] = dict(er=eager_ret, cr=comp_ret, ep=eager_peak, cp=comp_peak)
    print(f"{impl:>8} {eager_ret / MB:12.2f} MB {comp_ret / MB:11.2f} MB "
          f"{eager_peak / MB:9.2f} MB {comp_peak / MB:8.2f} MB   ({WARMUP} warm-ups, {warm:.0f} s)",
          flush=True)

print()
for key, label in (("er", "retained eager"), ("cr", "retained compiled"),
                   ("ep", "peak eager"), ("cp", "peak compiled")):
    v = {k: r[key] for k, r in rows.items()}
    print(f"{label:>20}: sparse/einsum = {v['sparse'] / v['einsum']:.4f}   "
          f"spread max/min = {max(v.values()) / min(v.values()):.4f}")
