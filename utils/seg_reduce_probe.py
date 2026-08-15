"""Gate-day probe for Phase 2.2b (sorted-segment aggregation): is torch.segment_reduce
a safe drop-in for index_add_ on THIS torch build?

Run INSIDE the campaign container (no repo imports, torch-only -- works on any box):

    python utils/seg_reduce_probe.py

Prints four verdicts; 2.2b may only be adopted if ALL four hold here AND the gate-day
profile still ranks the data scatter after the 2.2a degree hoist (docs/cgenn-compile.md,
"The improvement program", Phase 2.2b):

  1. BIT-FWD : segment_reduce == index_add_ bitwise on sorted ids (CPU reference class;
               on CUDA index_add_ is nondeterministic, so there the comparison is
               segment_reduce-vs-itself across repeats = determinism, plus TOL vs CPU).
  2. BIT-BWD : gradients bitwise-equal the same way.
  3. EMPTIES : zero-length segments (padded nodes -- guaranteed in every real batch)
               produce exactly 0.0, not NaN or a reduction identity.
  4. COMPILE : torch.compile(dynamic=True) traces it at 1 graph / 0 breaks and the
               compiled backward produces finite grads (lengths passed as a TENSOR,
               exactly how adoption would feed it from the eager edge hoist -- never
               an in-graph bincount).

Already verified green on torch 2.13.0 CPU and public 2.8.0+cu128 CPU (2026-08-15);
this file exists so the NGC-fork/Triton combination gets the same check on the card.
On CUDA, verdicts 1-2 additionally report run-to-run determinism of segment_reduce
itself (3 repeats), which is the whole point of 2.2b.
"""
import torch

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {dev}")

N, C, E = 640, 200, 15000
pool = torch.arange(2, N - 2, 2, device=dev)
ids = torch.sort(pool[torch.randint(0, pool.numel(), (E,), device=dev)]).values
lengths = torch.bincount(ids, minlength=N)
n_empty = int((lengths == 0).sum())
assert n_empty > 300, "probe must exercise many empty segments"

data = torch.randn(E, C, device=dev, requires_grad=True)
data2 = data.detach().clone().requires_grad_(True)

ref = data.new_zeros((N, C)).index_add_(0, ids, data)
got = torch.segment_reduce(data2, "sum", lengths=lengths, axis=0)
w = torch.linspace(-1, 1, ref.numel(), device=dev).view(ref.shape)
g_ref = torch.autograd.grad((ref * w).sum(), data)[0]
g_got = torch.autograd.grad((got * w).sum(), data2)[0]

fwd_bit = torch.equal(ref, got)
bwd_bit = torch.equal(g_ref, g_got)
rel = ((ref - got).abs().max() / ref.abs().max()).item()
fwd_msg = ("PASS (bitwise)" if fwd_bit else
           f"not bitwise, rel {rel:.1e} (expected on CUDA: index_add_ is the "
           "nondeterministic side)")
print(f"1. BIT-FWD : {fwd_msg}")
print(f"2. BIT-BWD : {'PASS (bitwise)' if bwd_bit else 'not bitwise vs index_add_'}")
print(f"3. EMPTIES : {'PASS' if bool((got[lengths == 0] == 0).all()) else 'FAIL'} "
      f"({n_empty} zero-length segments, all exactly 0.0)")

if dev == "cuda":
    reps = [torch.segment_reduce(data.detach(), "sum", lengths=lengths, axis=0)
            for _ in range(3)]
    det = all(torch.equal(reps[0], r) for r in reps[1:])
    print(f"   CUDA determinism of segment_reduce (3 repeats): {'PASS' if det else 'FAIL'}")

import torch._dynamo as dynamo


def agg(d, l):
    return torch.segment_reduce(d, "sum", lengths=l, axis=0)


ex = dynamo.explain(agg)(torch.randn(12000, 64, device=dev),
                         torch.bincount(torch.randint(0, 500, (12000,), device=dev),
                                        minlength=500))
try:
    f = torch.compile(agg, dynamic=True)
    dd = torch.randn(12000, 64, device=dev, requires_grad=True)
    ll = torch.bincount(torch.randint(0, 500, (12000,), device=dev), minlength=500)
    f(dd, ll).sum().backward()
    bwd_ok = bool(torch.isfinite(dd.grad).all())
except Exception as e:  # noqa: BLE001 -- the verdict IS whether this raises
    bwd_ok = f"FAILED: {type(e).__name__}: {e}"
print(f"4. COMPILE : graphs={ex.graph_count} breaks={ex.graph_break_count}, "
      f"compiled fwd+bwd finite grads: {bwd_ok}")
verdict = fwd_bit and bwd_bit if dev == "cpu" else True
print("VERDICT:", "2.2b SAFE on this build" if (ex.graph_break_count == 0 and bwd_ok is True)
      else "2.2b NOT safe here -- keep index_add_, see docstring")
